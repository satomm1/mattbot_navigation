"""Which objects to go and check, and the value/cost of each observation option.

Pure Python (no ROS). An observation option is "stop at (x, y), look at these objects".
Every option carries a kind, an estimated cost (extra seconds) and a value per object,
so policies can trade value against cost:

  OPPORTUNISTIC (implemented): the stop is on the planned path; cost = turn out + dwell + turn back.
  DETOUR (future): the stop is off the path; cost also includes the extra driving time.

ThresholdPolicy (current) checks objects whose belief is at or below a threshold. A future
DetourPolicy can add DETOUR options and gate on value - lambda * cost, with value using
importance() (e.g. how badly the object blocks a critical passage).
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

    @property
    def path_index(self):
        return self.stop.path_index if self.kind == OPPORTUNISTIC else -1


def importance(candidate):
    """How much it matters to know whether this object still blocks. Stub: all objects equal.

    Future: derive from map connectivity, e.g. the increase in shortest-path lengths if the
    object's cells are blocked (an object in a critical doorway matters more).
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
    ):
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

    def value(self, candidate):
        return importance(candidate) * (1.0 - candidate.belief)

    def options(self, path_xy, candidates, viewsheds, now, exclude=()):
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
        return options
