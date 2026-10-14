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

A detour is never taken if its route branches off the path within skip_start_m of the start or
skip_goal_m of the goal (DetourResult.branch_arc_m): such a check belongs to the trip that starts or
ends there, not to a detour from this one.

Scouts (SCOUT): a robot with no goal of its own leaves only to check objects (navigation_utils/scout.py:
drive from the robot to the best viewpoint). Same value V as a detour (own N and q,
ScoutValueParams); the cost C comes from a pluggable ScoutCostModel (default TravelTimeScoutCost:
out-and-back driving + turning and looking + opportunity cost of the time away). A scout is worth sending if
V - C > margin. Only the decision exists so far; nothing drives a scout yet.

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
SCOUT = 2

# How the navigator continues after the observation
REMAINING_PATH = 0  # stop is on the current path: continue its remaining trajectory
REPLAN = 1  # stop is off the path: replan to the goal from the stop
RETURN = 2  # scout: no goal to resume (placeholder until scouts are executed)


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
    detour_m: float = 0.0  # DETOUR: extra driving distance; SCOUT: start -> viewpoint
    value_m: float = 0.0  # DETOUR / SCOUT: V, expected future driving saved (sum over targets)
    cost_m: float = 0.0  # DETOUR / SCOUT: C, cost of the check now
    leave_xy: tuple = ()  # DETOUR: where the robot leaves the path; SCOUT: the start

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
    skip_start_m: float = 1.0  # no detour whose route branches off the path this close to its start
    skip_goal_m: float = 1.0  # ... or this close to its goal (arc length along the path)


@dataclass
class ScoutValueParams:
    """Scout rule (see the module docstring)."""

    n_trips: float = 5.0  # N: fleet trips that would keep avoiding a removed object
    conclusive_prob: float = 1.0  # q: probability a check gives PRESENT / ABSENT
    margin_m: float = 0.0  # scout only if V - C exceeds this
    max_scouts: int = 1  # options chosen per evaluation
    hard_cap_m: float = 30.0  # never search for viewpoints further than this (one way)


@dataclass
class ObservationEvaluation:
    """V and C of one possible DETOUR or SCOUT (also for the ones not taken, for logging)."""

    object_ids: List[str]
    detour_m: float  # DETOUR: extra driving; SCOUT: start -> viewpoint
    value_m: float
    cost_m: float
    chosen: bool = False
    option: object = None  # ObservationOption
    values_m: List[float] = field(default_factory=list)
    blocked: str = ""  # why it may not be taken whatever its value (e.g. "branches at the goal")
    kind: int = DETOUR
    source: object = None  # the detour.DetourResult / scout.ScoutResult it was built from
    breakdown: Dict[str, float] = field(default_factory=dict)  # SCOUT: terms of C (ScoutCostModel.breakdown)

    @property
    def net_m(self):
        return self.value_m - self.cost_m


DetourEvaluation = ObservationEvaluation


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


def sweep_turn(heading, x, y, targets):
    """Total turn (rad) from heading at (x, y) through targets (with .x, .y) in order; no turn back."""
    turn = 0.0
    for c in targets:
        bearing = math.atan2(c.y - y, c.x - x)
        turn += abs(wrap_angle(bearing - heading))
        heading = bearing
    return turn


class ScoutCostModel:
    """C of a scout (m). Pluggable: the observation planner picks one with ~scout_cost_model."""

    name = "base"

    def cost_m(self, result, targets, policy):
        """result: scout.ScoutResult; targets: Candidates in turn order; policy: ThresholdPolicy
        (cruise_speed, turn_rate, dwell_s)."""
        raise NotImplementedError

    def breakdown(self, result, targets, policy):
        """The terms of C (for logging and ScoutDecision); "total_m" is C."""
        return {"total_m": self.cost_m(result, targets, policy)}

    def travel_limit_m(self, value_m, policy):
        """Longest start -> viewpoint distance worth searching for an object of value value_m: the
        one-way distance at which even the cheapest scout (one target, no turn) costs value_m."""
        return value_m - policy.cruise_speed * policy.dwell_s


class TravelTimeScoutCost(ScoutCostModel):
    """Driving + looking + opportunity cost of being away (all in m; times at cruise_speed v):

        travel_m = to_view_m (+ return_m if round_trip)
        look_m   = v * (turn / turn_rate + dwell_s * n)       turn through the n targets from arrival
        T_away   = travel_m / v + turn / turn_rate + dwell_s * n                              (s)
        C        = travel_m + look_m + v * fixed_s + v * per_s_away * T_away

    fixed_s: cost of committing to a scout at all (s of driving). per_s_away: cost per second the
    robot is unavailable (s per s). Both 0 by default: C is then the physical cost alone.

    TODO: calibrate fixed_s and per_s_away (e.g. per_s_away from the mission rate), battery.
    """

    name = "travel_time"

    def __init__(self, round_trip=True, fixed_s=0.0, per_s_away=0.0):
        self.round_trip = bool(round_trip)
        self.fixed_s = float(fixed_s)
        self.per_s_away = float(per_s_away)

    def breakdown(self, result, targets, policy):
        v = policy.cruise_speed
        vx, vy = result.viewpoint_xy
        turn = sweep_turn(result.arrival_heading, vx, vy, targets)
        travel = result.to_view_m + (result.return_m if self.round_trip else 0.0)
        look_s = turn / policy.turn_rate + policy.dwell_s * len(targets)
        t_away = travel / v + look_s
        out = {
            "travel_m": travel,
            "look_m": v * look_s,
            "fixed_m": v * self.fixed_s,
            "away_m": v * self.per_s_away * t_away,
            "t_away_s": t_away,
        }
        out["total_m"] = out["travel_m"] + out["look_m"] + out["fixed_m"] + out["away_m"]
        return out

    def cost_m(self, result, targets, policy):
        return self.breakdown(result, targets, policy)["total_m"]

    def travel_limit_m(self, value_m, policy):
        # C_min(d) = (1 + per_s_away) * (k d + v dwell) + v fixed_s, k = 2 (round trip) or 1
        v = policy.cruise_speed
        k = 2.0 if self.round_trip else 1.0
        return ((value_m - v * self.fixed_s) / (1.0 + self.per_s_away) - v * policy.dwell_s) / k


SCOUT_COST_MODELS = {TravelTimeScoutCost.name: TravelTimeScoutCost}


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
        importance_m_fn=None,  # Candidate -> I_o (m per trip) or None if not evaluated (detours, scouts)
        detour_params=None,  # DetourValueParams
        scout_params=None,  # ScoutValueParams
        scout_cost=None,  # ScoutCostModel (default TravelTimeScoutCost)
    ):
        self.importance_fn = importance_fn
        self.importance_m_fn = importance_m_fn
        self.detour_params = detour_params or DetourValueParams()
        self.scout_params = scout_params or ScoutValueParams()
        self.scout_cost = scout_cost or TravelTimeScoutCost()
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
        vx, vy = detour.viewpoint_xy
        turn = sweep_turn(detour.arrival_heading, vx, vy, targets)
        return detour.detour_m + self.cruise_speed * (turn / self.turn_rate + self.dwell_s * len(targets))

    def detour_blocked(self, detour):
        """Reason a detour may not be taken (branches off at the start or goal), or ""."""
        p = self.detour_params
        if detour.branch_arc_m < p.skip_start_m:
            return "branches at the start"
        if detour.path_length_m - detour.branch_arc_m < p.skip_goal_m:
            return "branches at the goal"
        return ""

    def detour_options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None):
        """Detours worth driving. Returns (chosen ObservationOptions, all DetourEvaluations).

        One evaluation per object with a detour result: its best viewpoint, bundled with every
        other detour candidate visible from there (values add). Options with V - C > margin are
        taken best-first by net benefit, without sharing objects, up to max_detours_per_path.
        Blocked ones (detour_blocked: branching off at the start or goal) are evaluated but never taken.
        """
        detours = detours or {}
        cands = {c.object_id: c for c in self.detour_candidates(path_xy, candidates, viewsheds, now, exclude)}
        evaluations = []
        for object_id, c in cands.items():
            d = detours.get(object_id)
            if d is None:
                continue
            vx, vy = d.viewpoint_xy
            targets = self._bundle(c, cands, viewsheds, vx, vy, d.arrival_heading)
            values = [self.detour_value_m(u) for u in targets]
            cost_m = self.detour_cost_m(d, targets)
            stop = Stop(d.leave_index, vx, vy, d.arrival_heading, [(u.object_id, u.x, u.y) for u in targets])
            option = ObservationOption(
                kind=DETOUR, resume=REPLAN, stop=stop, candidates=targets, values=values,
                cost_s=cost_m / self.cruise_speed, detour_m=d.detour_m, value_m=sum(values), cost_m=cost_m,
                leave_xy=tuple(d.leave_xy),
            )
            evaluations.append(ObservationEvaluation([u.object_id for u in targets], d.detour_m, sum(values),
                                                     cost_m, option=option, values_m=values,
                                                     blocked=self.detour_blocked(d), source=d))
        p = self.detour_params
        chosen = self._choose(evaluations, p.max_detours_per_path, p.margin_m, self.max_cost_s)
        return chosen, evaluations

    # ---------- Scouts ----------

    def covered_elsewhere(self, candidate):
        """True if the object will be checked without a scout (stub: always False).

        TODO: a peer's planned path passes one of its viewpoints, or another robot has claimed it.
        """
        return False

    def scout_eligible(self, candidate, now, exclude=()):
        """As detour_eligible, and not covered elsewhere (no belief threshold)."""
        return self.detour_eligible(candidate, now, exclude) and not self.covered_elsewhere(candidate)

    def scout_candidates(self, candidates, viewsheds, now, exclude=()):
        """Eligible objects with a viewshed."""
        return [c for c in candidates if c.object_id in viewsheds and self.scout_eligible(c, now, exclude)]

    def scout_value_m(self, candidate):
        p = self.scout_params
        return detour_value_m(self.importance_m_fn(candidate), candidate.belief, p.n_trips, p.conclusive_prob)

    def max_scout_m(self, candidate):
        """Longest start -> viewpoint distance that could pay off for this object alone (<= 0: never)."""
        limit = self.scout_cost.travel_limit_m(self.scout_value_m(candidate), self)
        return min(limit, self.scout_params.hard_cap_m)

    def scout_options(self, start_xy, candidates, viewsheds, now, exclude=(), scouts=None):
        """Scouts worth sending from start_xy. Returns (chosen ObservationOptions, all
        ObservationEvaluations).

        scouts: {object_id: scout.ScoutResult} for scout_candidates(). One evaluation per object
        with a result: its nearest viewpoint, bundled with every other candidate visible from there
        (values add). Options with V - C > margin are taken best-first by net benefit, without
        sharing objects, up to max_scouts.
        """
        scouts = scouts or {}
        cands = {c.object_id: c for c in self.scout_candidates(candidates, viewsheds, now, exclude)}
        evaluations = []
        for object_id, c in cands.items():
            r = scouts.get(object_id)
            if r is None:
                continue
            vx, vy = r.viewpoint_xy
            targets = self._bundle(c, cands, viewsheds, vx, vy, r.arrival_heading)
            values = [self.scout_value_m(u) for u in targets]
            parts = self.scout_cost.breakdown(r, targets, self)
            cost_m = parts["total_m"]
            stop = Stop(-1, vx, vy, r.arrival_heading, [(u.object_id, u.x, u.y) for u in targets])
            option = ObservationOption(
                kind=SCOUT, resume=RETURN, stop=stop, candidates=targets, values=values,
                cost_s=cost_m / self.cruise_speed, detour_m=r.to_view_m, value_m=sum(values), cost_m=cost_m,
                leave_xy=tuple(r.start_xy),
            )
            evaluations.append(ObservationEvaluation([u.object_id for u in targets], r.to_view_m, sum(values),
                                                     cost_m, option=option, values_m=values, kind=SCOUT,
                                                     source=r, breakdown=parts))
        p = self.scout_params
        return self._choose(evaluations, p.max_scouts, p.margin_m), evaluations

    # ---------- Shared by detours and scouts ----------

    @staticmethod
    def _bundle(c, cands, viewsheds, vx, vy, heading):
        """c plus every other candidate visible from (vx, vy), swept in one direction from heading."""
        targets = [c] + [
            u for oid, u in cands.items()
            if oid != c.object_id and viewsheds[oid].contains_xy([(vx, vy)])[0]
        ]
        targets.sort(key=lambda u: wrap_angle(math.atan2(u.y - vy, u.x - vx) - heading))
        return targets

    @staticmethod
    def _choose(evaluations, max_n, margin_m, max_cost_s=0.0):
        """Mark and return the options with V - C > margin_m, best net first, no object shared,
        not blocked, at most max_n (and cost_s <= max_cost_s if > 0)."""
        chosen, used = [], set()
        for ev in sorted(evaluations, key=lambda e: -e.net_m):
            if len(chosen) >= max_n or ev.net_m <= margin_m:
                break
            if used & set(ev.object_ids) or ev.blocked:
                continue
            if max_cost_s > 0 and ev.option.cost_s > max_cost_s:
                continue
            ev.chosen = True
            chosen.append(ev.option)
            used |= set(ev.object_ids)
        return chosen
