"""Which objects to go and check, and the value/cost of each observation option.

Pure Python (no ROS). An observation option is "stop at (x, y), look at these objects".
Every option carries a kind, an estimated cost (extra seconds) and a value per object,
so policies can trade value against cost:

  OPPORTUNISTIC: the stop is on the planned path; cost = turn out + dwell + turn back.
  DETOUR: the stop is a viewpoint off the path (navigation_utils/detour.py: leave the path, look,
          replan to the goal).

Opportunistic stops (ThresholdPolicy): objects with belief <= check_below_belief that are visible
from the path. Value = importance * (1 - belief), importance from importance_fn (the observation
planner passes I_o) or the importance() stub.

Detours (value of information): only for objects that are NOT visible from any point of the path
(those are left to opportunistic stops, whatever their belief). A detour pays off if the expected
future driving it saves exceeds what it costs now, both in metres:

    p_gone = (1 - b) / 2                      belief b in [0, 1]: probability the object has moved
    V      = q * N * p_gone * I_o             N trips would keep avoiding it; q = P(check conclusive)
    C      = D + v_cruise * (t_turn + t_dwell) extra driving + time spent turning and looking
    detour if V - C > margin;   D* = V - v_cruise * t_dwell is the longest detour worth searching

Only "the fleet keeps avoiding an object that has actually gone" is modelled (the fleet routes
around every ledger object until it is removed), so the discovery cost I_o_disc is not used.
"""

import math
from dataclasses import dataclass, field
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
    value_m: float = 0.0  # DETOUR: V, expected future driving saved (sum over targets)
    cost_m: float = 0.0  # DETOUR: C, cost of the check now
    leave_xy: tuple = ()  # DETOUR: where the robot leaves the path

    @property
    def net_m(self):
        return self.value_m - self.cost_m

    @property
    def path_index(self):
        """Index into the request path: the stop (OPPORTUNISTIC) or the leave point (DETOUR)."""
        return self.stop.path_index


@dataclass
class DetourValueParams:
    """Value-of-information detour rule (see the module docstring)."""

    n_trips: float = 5.0  # N: fleet trips that would keep avoiding a removed object
    conclusive_prob: float = 1.0  # q: probability a check gives PRESENT / ABSENT
    margin_m: float = 0.0  # detour only if V - C exceeds this
    max_detours_per_path: int = 1
    hard_cap_m: float = 15.0  # never search for detours longer than this


@dataclass
class DetourEvaluation:
    """V and C of one possible detour (also for the ones not taken, for logging)."""

    object_ids: List[str]
    detour_m: float
    value_m: float
    cost_m: float
    chosen: bool = False
    option: object = None  # ObservationOption
    values_m: List[float] = field(default_factory=list)

    @property
    def net_m(self):
        return self.value_m - self.cost_m


def p_gone(belief):
    """Probability that a ledger object with belief b in [0, 1] has been moved: (1 - b) / 2."""
    return (1.0 - min(max(float(belief), 0.0), 1.0)) / 2.0


def detour_value_m(importance_m, belief, n_trips, conclusive_prob=1.0):
    """V: expected future driving saved by checking (m)."""
    return conclusive_prob * n_trips * p_gone(belief) * max(float(importance_m), 0.0)


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
        max_turn=math.pi / 2,  # rad: prefer stops that see the object within this turn (None = no preference)
        turn_rate=1.0,  # rad/s, for the cost estimate
        dwell_s=3.0,
        importance_fn=None,  # Candidate -> importance; None = importance() stub
        cruise_speed=0.4,  # m/s, converts time to distance for detours
        importance_m_fn=None,  # Candidate -> I_o (m per trip) or None if not evaluated (detours)
        detour_params=None,  # DetourValueParams
    ):
        self.importance_fn = importance_fn
        self.importance_m_fn = importance_m_fn
        self.detour_params = detour_params or DetourValueParams()
        self.cruise_speed = cruise_speed
        self.check_below_belief = check_below_belief
        self.cooldown_s = cooldown_s
        self.max_cost_s = max_cost_s
        self.r_pref = r_pref
        self.skip_start_m = skip_start_m
        self.skip_goal_m = skip_goal_m
        self.merge_m = merge_m
        self.max_turn = max_turn
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

    # ---------- Opportunistic ----------

    def options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None):
        """Opportunistic options along path_xy, then the chosen DETOUR options (detours:
        {object_id: detour.DetourResult} for detour_candidates(); None = no detours)."""
        options = self.opportunistic_options(path_xy, candidates, viewsheds, now, exclude)
        if detours is not None:
            options += self.detour_options(path_xy, candidates, viewsheds, now, exclude, detours)[0]
        return options

    def opportunistic_options(self, path_xy, candidates, viewsheds, now, exclude=()):
        eligible = {c.object_id: c for c in self.eligible_candidates(candidates, now, exclude)}
        stops = select_stops(
            path_xy,
            [(c.object_id, c.x, c.y) for c in eligible.values()],
            viewsheds,
            r_pref=self.r_pref,
            skip_start_m=self.skip_start_m,
            skip_goal_m=self.skip_goal_m,
            merge_m=self.merge_m,
            max_turn=self.max_turn,
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

    # ---------- Detours ----------

    def detour_eligible(self, candidate, now, exclude=()):
        """Not excluded, out of cooldown, and with a known importance (no belief threshold)."""
        if candidate.object_id in exclude:
            return False
        last = self.last_checked.get(candidate.object_id)
        if last is not None and now - last < self.cooldown_s:
            return False
        return self.importance_m_fn is not None and self.importance_m_fn(candidate) is not None

    @staticmethod
    def checkable_from_path(path_xy, viewshed):
        """Visible from any point of the path (incl. near start / goal): an opportunistic job."""
        return bool(viewshed.contains_xy(path_xy).any())

    def detour_candidates(self, path_xy, candidates, viewsheds, now, exclude=()):
        """Eligible objects that no point of the path can see: the only ones a detour may serve."""
        out = []
        for c in candidates:
            vs = viewsheds.get(c.object_id)
            if vs is None or not self.detour_eligible(c, now, exclude):
                continue
            if not self.checkable_from_path(path_xy, vs):
                out.append(c)
        return out

    def detour_value_m(self, candidate):
        p = self.detour_params
        return detour_value_m(self.importance_m_fn(candidate), candidate.belief, p.n_trips, p.conclusive_prob)

    def max_detour_m(self, candidate):
        """D*: the longest detour that could pay off for this object alone (<= 0: never)."""
        d_star = self.detour_value_m(candidate) - self.cruise_speed * self.dwell_s
        return min(d_star, self.detour_params.hard_cap_m)

    def detour_cost_m(self, detour, targets):
        """C: extra driving + cruise_speed * (turn through the targets from the arrival heading
        + dwell per target). No turn back: the robot replans from the viewpoint."""
        heading, turn = detour.arrival_heading, 0.0
        for c in targets:
            bearing = math.atan2(c.y - detour.viewpoint_xy[1], c.x - detour.viewpoint_xy[0])
            turn += abs(wrap_angle(bearing - heading))
            heading = bearing
        return detour.detour_m + self.cruise_speed * (turn / self.turn_rate + self.dwell_s * len(targets))

    def detour_options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None):
        """Detours worth driving. Returns (chosen ObservationOptions, all DetourEvaluations).

        One evaluation per object with a detour result: its best viewpoint, bundled with every
        other detour candidate visible from there (values add). Options with V - C > margin are
        taken best-first by net benefit, without sharing objects, up to max_detours_per_path.
        """
        detours = detours or {}
        cands = {c.object_id: c for c in self.detour_candidates(path_xy, candidates, viewsheds, now, exclude)}
        evaluations = []
        for object_id, c in cands.items():
            d = detours.get(object_id)
            if d is None:
                continue
            vx, vy = d.viewpoint_xy
            targets = [c] + [
                u for oid, u in cands.items()
                if oid != object_id and viewsheds[oid].contains_xy([(vx, vy)])[0]
            ]
            # Sweep the targets in one direction from the arrival heading
            targets.sort(key=lambda u: wrap_angle(math.atan2(u.y - vy, u.x - vx) - d.arrival_heading))
            values = [self.detour_value_m(u) for u in targets]
            cost_m = self.detour_cost_m(d, targets)
            stop = Stop(d.leave_index, vx, vy, d.arrival_heading, [(u.object_id, u.x, u.y) for u in targets])
            option = ObservationOption(
                kind=DETOUR, resume=REPLAN, stop=stop, candidates=targets, values=values,
                cost_s=cost_m / self.cruise_speed, detour_m=d.detour_m, value_m=sum(values), cost_m=cost_m,
                leave_xy=tuple(d.leave_xy),
            )
            evaluations.append(DetourEvaluation([u.object_id for u in targets], d.detour_m, sum(values), cost_m,
                                                option=option, values_m=values))

        p = self.detour_params
        chosen, used = [], set()
        for ev in sorted(evaluations, key=lambda e: -e.net_m):
            if len(chosen) >= p.max_detours_per_path or ev.net_m <= p.margin_m:
                break
            if used & set(ev.object_ids):
                continue
            if self.max_cost_s > 0 and ev.option.cost_s > self.max_cost_s:
                continue
            ev.chosen = True
            chosen.append(ev.option)
            used |= set(ev.object_ids)
        return chosen, evaluations
