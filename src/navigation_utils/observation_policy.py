"""Which objects to go and check, and the value/cost of each observation option.

Pure Python (no ROS). An observation option is "stop at (x, y), look at these objects".
Every option carries a kind, an estimated cost (extra seconds) and a value per object,
so policies can trade value against cost:

  OPPORTUNISTIC: the stop is on the planned path; cost = turn out + dwell + turn back.
  DETOUR: the stop is a viewpoint off the path (navigation_utils/detour.py: leave the path, look,
          replan to the goal); cost = extra driving / cruise_speed + turn to the object + dwell.
          Offered only for objects that no opportunistic stop covers. The navigator does not
          execute detours yet.

ThresholdPolicy (current) checks objects whose belief is at or below a threshold. Its value is
importance * (1 - belief); importance comes from importance_fn when given (the observation
planner passes the obstacle importance I_o from obstacle_importance.py, metres of extra travel
per trip), else the importance() stub. A future DetourPolicy can add DETOUR options and gate on
value - lambda * cost, using I_o (or the discovery cost I_o_disc) as the importance.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

from navigation_utils.viewpoints import Stop, select_stops, wrap_angle

# Option kinds
OPPORTUNISTIC = 0
DETOUR = 1

# How the navigator continues after the observation
REMAINING_PATH = 0  # stop is on the current path: continue its remaining trajectory
REPLAN = 1  # stop is off the path: replan to the goal from the stop


@dataclass
class Candidate:
    """An object that might be checked (from /object_beliefs; position in the local map frame)."""

    object_id: str
    class_name: str
    x: float
    y: float
    belief: float
    width: float = 0.5  # m; known objects also block line of sight to other objects


@dataclass
class ObservationOption:
    kind: int
    resume: int
    stop: Stop
    candidates: List[Candidate]  # in stop.targets order
    values: List[float]
    cost_s: float
    detour_m: float = 0.0  # extra driving distance (DETOUR only)

    @property
    def path_index(self):
        return self.stop.path_index if self.kind == OPPORTUNISTIC else -1


def importance(candidate):
    """Default importance when no importance_fn is given: all objects equal.

    The map-connectivity importance I_o (extra travel per trip if the object blocks) is in
    navigation_utils/obstacle_importance.py and is passed to policies as importance_fn.
    """
    return 1.0


def opportunistic_cost_s(stop, turn_rate, dwell_s):
    """Extra time for a stop on the path: turn to each target in order, dwell, turn back."""
    heading = stop.heading
    total_turn = 0.0
    for _object_id, tx, ty in stop.targets:
        bearing = math.atan2(ty - stop.y, tx - stop.x)
        total_turn += abs(wrap_angle(bearing - heading))
        heading = bearing
    total_turn += abs(wrap_angle(stop.heading - heading))
    return total_turn / turn_rate + dwell_s * len(stop.targets)


class ObservationPolicy:
    """Base policy: decides which objects are worth checking and builds options for a path."""

    def eligible(self, candidate, now, exclude=()):
        raise NotImplementedError

    def value(self, candidate):
        raise NotImplementedError

    def eligible_candidates(self, candidates, now, exclude=()):
        return [c for c in candidates if self.eligible(c, now, exclude)]

    def options(self, path_xy, candidates, viewsheds, now, exclude=()):
        raise NotImplementedError


class ThresholdPolicy(ObservationPolicy):
    """Opportunistically check objects with belief <= check_below_belief.

    Each object is checked at most once per call (i.e. per path); the caller passes objects
    already checked for the current goal in `exclude`. After a check, an object is not
    eligible again for cooldown_s (perception may not have resolved it yet).
    """

    def __init__(
        self,
        check_below_belief=0.5,
        cooldown_s=120.0,
        max_cost_s=0.0,  # 0 = no cost limit per stop
        r_pref=2.0,
        skip_start_m=0.5,
        skip_goal_m=0.8,
        merge_m=0.5,
        turn_rate=1.0,  # rad/s, for the cost estimate
        dwell_s=3.0,
        importance_fn=None,  # Candidate -> importance; None = importance() stub
        cruise_speed=0.4,  # m/s, converts detour distance to time
    ):
        self.importance_fn = importance_fn
        self.cruise_speed = cruise_speed
        self.check_below_belief = check_below_belief
        self.cooldown_s = cooldown_s
        self.max_cost_s = max_cost_s
        self.r_pref = r_pref
        self.skip_start_m = skip_start_m
        self.skip_goal_m = skip_goal_m
        self.merge_m = merge_m
        self.turn_rate = turn_rate
        self.dwell_s = dwell_s
        self.last_checked: Dict[str, float] = {}  # object_id -> time of last completed check

    def record_check(self, object_id, t):
        self.last_checked[object_id] = max(t, self.last_checked.get(object_id, t))

    def eligible(self, candidate, now, exclude=()):
        if candidate.object_id in exclude or candidate.belief > self.check_below_belief:
            return False
        last = self.last_checked.get(candidate.object_id)
        return last is None or now - last >= self.cooldown_s

    def importance(self, candidate):
        return self.importance_fn(candidate) if self.importance_fn is not None else importance(candidate)

    def value(self, candidate):
        return self.importance(candidate) * (1.0 - candidate.belief)

    def detour_cost_s(self, detour, candidate):
        """Extra time for a DETOUR: drive the extra distance, turn to the object, dwell."""
        bearing = math.atan2(candidate.y - detour.viewpoint_xy[1], candidate.x - detour.viewpoint_xy[0])
        turn = abs(wrap_angle(bearing - detour.arrival_heading))
        return detour.detour_m / self.cruise_speed + turn / self.turn_rate + self.dwell_s

    def options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None):
        """Opportunistic options along path_xy, then DETOUR options for eligible objects that no
        opportunistic stop covers (detours: {object_id: detour.DetourResult}, None = none)."""
        eligible = {c.object_id: c for c in self.eligible_candidates(candidates, now, exclude)}
        stops = select_stops(
            path_xy,
            [(c.object_id, c.x, c.y) for c in eligible.values()],
            viewsheds,
            r_pref=self.r_pref,
            skip_start_m=self.skip_start_m,
            skip_goal_m=self.skip_goal_m,
            merge_m=self.merge_m,
        )
        options = []
        for stop in stops:
            cost = opportunistic_cost_s(stop, self.turn_rate, self.dwell_s)
            if self.max_cost_s > 0 and cost > self.max_cost_s:
                continue
            cands = [eligible[object_id] for object_id, _x, _y in stop.targets]
            options.append(
                ObservationOption(
                    kind=OPPORTUNISTIC,
                    resume=REMAINING_PATH,
                    stop=stop,
                    candidates=cands,
                    values=[self.value(c) for c in cands],
                    cost_s=cost,
                )
            )
        covered = {c.object_id for opt in options for c in opt.candidates}
        detour_options = []
        for object_id, c in eligible.items():
            d = (detours or {}).get(object_id)
            if object_id in covered or d is None or d.on_path:
                continue
            cost = self.detour_cost_s(d, c)
            if self.max_cost_s > 0 and cost > self.max_cost_s:
                continue
            stop = Stop(d.leave_index, d.viewpoint_xy[0], d.viewpoint_xy[1], d.arrival_heading,
                        [(object_id, c.x, c.y)])
            detour_options.append(
                ObservationOption(kind=DETOUR, resume=REPLAN, stop=stop, candidates=[c],
                                  values=[self.value(c)], cost_s=cost, detour_m=d.detour_m)
            )
        detour_options.sort(key=lambda o: o.cost_s)
        return options + detour_options
