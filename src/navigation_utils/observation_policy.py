"""Which objects to go and check, and the value/cost of each observation option.

Pure Python (no ROS). An observation option is "stop at (x, y), look at these objects".
Every option carries a kind, an estimated cost (extra seconds) and a value per object,
so policies can trade value against cost:

  OPPORTUNISTIC: the stop is on the planned path; cost = turn out + dwell + turn back.
  AT_START / AT_GOAL: the robot looks while it is stopped anyway: from its start pose before leaving
          (first plan for a goal only) or from the park pose on arrival; cost = turn + dwell.
  DETOUR: the stop is a viewpoint off the path (navigation_utils/detour.py: leave the path, look,
          replan to the goal).

Opportunistic stops (ThresholdPolicy): objects with belief <= check_below_belief that are visible
from the path. Value = importance * (1 - belief), importance from importance_fn (the observation
planner passes I_o) or the importance() stub. On-path stops are never placed within skip_start_m /
skip_goal_m of the start / goal; an eligible object that gets no on-path stop (only visible there,
or its stop exceeds max_cost_s) is checked from the start pose (AT_START) or the park pose (AT_GOAL)
if it is visible from there.

Detours (value of information): only for objects that no opportunistic or endpoint stop serves. An
object is left to those stops if it has one now, or if its belief is still above check_below_belief
and some point they may use (a usable path point, the start pose if allowed, the goal) sees it: it
gets a stop once its belief has decayed. A detour pays off if the expected future driving it saves
exceeds what it costs now, both in metres:

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

import numpy as np

from navigation_utils.viewpoints import Stop, path_headings, select_stops, usable_points, wrap_angle

# Option kinds (mattbot_dds/ObservationStop)
OPPORTUNISTIC = 0
DETOUR = 1
SCOUT = 2
AT_START = 3
AT_GOAL = 4

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
    last_stamp: float = None  # latest observation (ROS time); None = belief cannot be predicted (stays `belief`)


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
    to_view_s: float = 0.0  # DETOUR: leave point -> viewpoint; SCOUT: start -> viewpoint (s at cruise speed)

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


def covered_ids(options):
    """Object ids served by options (ObservationOptions)."""
    return frozenset(c.object_id for opt in options for c in opt.candidates)


def object_belief(age_s, hold_s, decay_s):
    """Belief that an object last observed age_s ago is still there: 1 for hold_s, then linear decay to
    0 over decay_s (as mattbot_dds dds_utils/belief.py object_belief; copied to avoid the dependency)."""
    if age_s <= hold_s:
        return 1.0
    if decay_s <= 0 or age_s >= hold_s + decay_s:
        return 0.0
    return 1.0 - (age_s - hold_s) / decay_s


def check_etas(stops, times):
    """Seconds from now until the check of each stop (in the given order).

    stops: ObservationOptions or mattbot_dds ObservationStop msgs (kind, path_index, cost_s, to_view_s);
    times: trajectory time (s from now) at each request path point. Stops are checked in path order
    (AT_START first, AT_GOAL last); each one delays the later ones by its cost_s:
        AT_START 0;  on-path times[i] + delay;  DETOUR times[leave] + delay + to_view_s;  AT_GOAL times[-1] + delay
    """
    times = np.asarray(times, dtype=float)
    last = len(times) - 1

    def order(k):
        s = stops[k]
        if s.kind == AT_START:
            return (-1, 0, k)
        if s.kind == AT_GOAL:
            return (last + 1, 0, k)
        return (int(s.path_index), int(s.kind == DETOUR), k)

    etas = [0.0] * len(stops)
    delay = 0.0
    for k in sorted(range(len(stops)), key=order):
        s = stops[k]
        if s.kind == AT_START:
            eta = 0.0
        elif s.kind == AT_GOAL:
            eta = float(times[last]) + delay
        else:
            eta = float(times[min(max(int(s.path_index), 0), last)]) + delay
            if s.kind == DETOUR:
                eta += float(getattr(s, "to_view_s", 0.0))
        etas[k] = eta
        delay += float(s.cost_s)
    return etas


def yields_to(kind, my_eta, my_id, intents, margin_s=10.0, commit_s=15.0, committed=False, remaining_s=None):
    """The other robot's planned check of the same object that this one gives way to, or None.

    intents: other robots' intents for the object (.robot_id, .kind, .eta, .committed). Earliest check wins:
      DETOUR: never yields once committed with remaining_s < commit_s (nearly there). Otherwise yields to a
              check earlier by more than margin_s; within the margin a committed check beats an uncommitted
              one, then the lower robot id wins.
      SCOUT:  the most expensive check: yields to any non-scout check that is not later by more than
              margin_s; against another scout, as a detour (earlier wins, the lower id within the margin).
      Opportunistic and endpoint checks never yield (None).
    """
    if kind not in (DETOUR, SCOUT):
        return None
    if kind == DETOUR and committed and remaining_s is not None and remaining_s < commit_s:
        return None
    winners = []
    for o in intents:
        if o.robot_id == my_id:
            continue
        if kind == SCOUT and o.kind != SCOUT:
            wins = o.eta <= my_eta + margin_s
        elif o.eta < my_eta - margin_s:
            wins = True
        elif abs(o.eta - my_eta) <= margin_s:
            if kind == DETOUR and bool(o.committed) != bool(committed):
                wins = bool(o.committed)
            else:
                wins = o.robot_id < my_id
        else:
            wins = False
        if wins:
            winners.append(o)
    return min(winners, key=lambda o: (o.eta, o.robot_id)) if winners else None


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


def endpoint_cost_s(x, y, heading_in, heading_out, targets, turn_rate, dwell_s):
    """Extra time for an AT_START / AT_GOAL stop at (x, y): turn from heading_in through targets (with
    .x, .y) in order, dwell at each, then turn to heading_out (the path's start heading / the goal heading).
    Only the part beyond the turn heading_in -> heading_out the robot makes anyway is extra."""
    if not targets:
        return 0.0
    turn = sweep_turn(heading_in, x, y, targets)
    last = math.atan2(targets[-1].y - y, targets[-1].x - x)
    turn += abs(wrap_angle(heading_out - last))
    turn -= abs(wrap_angle(heading_out - heading_in))
    return max(turn, 0.0) / turn_rate + dwell_s * len(targets)


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

    Time: with hold_s / decay_s set and candidates carrying last_stamp, every rule is evaluated at the
    time the check would happen (belief_at): a path request with `times` plans a stop wherever the
    object will be due when the robot gets there, and revalidate() re-applies the rules when it does.
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
        hold_s=None,  # belief model (mattbot_dds object_belief_map ~hold_s / ~decay_s); None = no prediction
        decay_s=None,
        robot_id=None,  # this robot, for fleet coordination (None: no coordination)
        intent_margin_s=10.0,  # yields_to margin
    ):
        self.hold_s = hold_s
        self.decay_s = decay_s
        self.robot_id = robot_id
        self.intent_margin_s = intent_margin_s
        self.peer_intents = []  # other robots' live intents (mattbot_dds /observation/intents), set by the planner
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

    # ---------- Belief and eligibility over time ----------

    def belief_at(self, candidate, t=None):
        """Belief at ROS time t if no new observation arrives (candidate.belief without a model or t)."""
        if t is None or candidate.last_stamp is None or self.hold_s is None or self.decay_s is None:
            return candidate.belief
        return object_belief(t - candidate.last_stamp, self.hold_s, self.decay_s)

    def in_cooldown(self, candidate, t):
        last = self.last_checked.get(candidate.object_id)
        return last is not None and t - last < self.cooldown_s

    def eligible_at(self, candidate, t, exclude=()):
        """Due an opportunistic / endpoint check at time t: not excluded, belief <= check_below_belief, no cooldown."""
        return (candidate.object_id not in exclude and not self.in_cooldown(candidate, t)
                and self.belief_at(candidate, t) <= self.check_below_belief)

    def eligible(self, candidate, now, exclude=()):
        return self.eligible_at(candidate, now, exclude)

    def t_due(self, candidate, now, exclude=()):
        """Earliest time >= now at which the object is eligible if nothing new is observed (inf: not without
        new information). Belief only falls and cooldowns only end, so it stays eligible from then on."""
        if candidate.object_id in exclude:
            return math.inf
        t = now
        last = self.last_checked.get(candidate.object_id)
        if last is not None:
            t = max(t, last + self.cooldown_s)
        if self.belief_at(candidate, t) <= self.check_below_belief:
            return t
        if candidate.last_stamp is None or self.hold_s is None or self.decay_s is None:
            return math.inf
        if self.decay_s <= 0:
            t_b = candidate.last_stamp + self.hold_s + 1e-6  # belief drops to 0 right after the hold
        else:
            t_b = candidate.last_stamp + self.hold_s + (1.0 - self.check_below_belief) * self.decay_s
        return max(t, t_b)

    def importance(self, candidate):
        return self.importance_fn(candidate) if self.importance_fn is not None else importance(candidate)

    def value(self, candidate, t=None):
        """importance * (1 - belief), with the belief at time t (now: candidate.belief)."""
        return self.importance(candidate) * (1.0 - self.belief_at(candidate, t))

    @staticmethod
    def _path_times(path_xy, times):
        """Trajectory time (s from now) per path point; zeros without times (everything decided for now)."""
        n = len(path_xy)
        if times is None or len(times) != n:
            return np.zeros(n)
        return np.asarray(times, dtype=float)

    # ---------- Opportunistic ----------

    def options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None, start_pose=None,
                goal_theta=None, times=None):
        """Opportunistic and endpoint options along path_xy, then the chosen DETOUR options (detours:
        {object_id: detour.DetourResult} for detour_candidates(); None = no detours)."""
        options = self.opportunistic_options(path_xy, candidates, viewsheds, now, exclude, start_pose, goal_theta,
                                             times)
        if detours is not None:
            options += self.detour_options(path_xy, candidates, viewsheds, now, exclude, detours,
                                           covered=covered_ids(options),
                                           start_xy=start_pose[:2] if start_pose is not None else None,
                                           times=times)[0]
        return options

    def opportunistic_options(self, path_xy, candidates, viewsheds, now, exclude=(), start_pose=None,
                              goal_theta=None, times=None):
        """On-path stops along path_xy, then at most one AT_START and one AT_GOAL stop for eligible
        objects that got no on-path stop.

        start_pose: (x, y, theta) of the robot when it may look before leaving (first plan for a goal),
        else None (no AT_START). goal_theta: final heading at the goal (default: the path's end heading).
        The AT_GOAL stop is at path_xy[-1], which must be the park pose.
        times: trajectory time (s from now) per path point. An object may get a stop wherever it will be
        due when the robot gets there (t_due); without times only objects due now are considered.
        """
        path_xy = np.asarray(path_xy, dtype=float)
        tt = self._path_times(path_xy, times)
        t_end = now + (float(tt[-1]) if len(tt) else 0.0)
        due = {}  # object_id -> (Candidate, t_due), due by the end of the path
        for c in candidates:
            td = self.t_due(c, now, exclude)
            if td <= t_end + 1e-9:
                due[c.object_id] = (c, td)
        eligible = {oid: c for oid, (c, _td) in due.items()}
        stops = select_stops(
            path_xy,
            [(c.object_id, c.x, c.y) for c in eligible.values()],
            viewsheds,
            r_pref=self.r_pref,
            skip_start_m=self.skip_start_m,
            skip_goal_m=self.skip_goal_m,
            merge_m=self.merge_m,
            max_turn=self.max_turn,
            due_masks={oid: now + tt >= td - 1e-9 for oid, (_c, td) in due.items() if td > now},
        )
        options = []
        for stop in stops:
            cost = opportunistic_cost_s(stop, self.turn_rate, self.dwell_s)
            if self.max_cost_s > 0 and cost > self.max_cost_s:
                continue
            cands = [eligible[object_id] for object_id, _x, _y in stop.targets]
            t_stop = now + float(tt[stop.path_index]) if times is not None else None
            options.append(
                ObservationOption(
                    kind=OPPORTUNISTIC,
                    resume=REMAINING_PATH,
                    stop=stop,
                    candidates=cands,
                    values=[self.value(c, t_stop) for c in cands],
                    cost_s=cost,
                )
            )
        due_at = {oid: td for oid, (_c, td) in due.items()}
        options += self.endpoint_options(path_xy, eligible, viewsheds, covered_ids(options), start_pose, goal_theta,
                                         due_at=due_at, now=now, t_end=t_end if times is not None else None)
        return options

    # ---------- Endpoints ----------

    def endpoint_options(self, path_xy, eligible, viewsheds, covered=(), start_pose=None, goal_theta=None,
                         due_at=None, now=None, t_end=None):
        """AT_START / AT_GOAL options for the eligible objects ({object_id: Candidate}) not in covered.

        An object visible from the start pose goes to AT_START (the robot is there now), else one
        visible from the park pose path_xy[-1] to AT_GOAL. A bundle over max_cost_s loses its
        lowest-value targets until it fits. due_at {object_id: t_due} with now / t_end (arrival time
        at the goal): AT_START only takes objects due now, AT_GOAL those due on arrival.
        """
        path_xy = np.asarray(path_xy, dtype=float)
        if len(path_xy) < 2:
            return []
        rest = [c for oid, c in eligible.items() if oid not in covered and oid in viewsheds]
        headings = path_headings(path_xy)
        gx, gy = float(path_xy[-1, 0]), float(path_xy[-1, 1])
        ends = []
        if start_pose is not None:
            sx, sy, sth = (float(v) for v in start_pose)
            ends.append((AT_START, 0, sx, sy, sth, float(headings[0]), now))
        g_out = float(goal_theta) if goal_theta is not None else float(headings[-1])
        ends.append((AT_GOAL, len(path_xy) - 1, gx, gy, float(headings[-1]), g_out, t_end))

        options = []
        for kind, index, x, y, heading_in, heading_out, t_check in ends:
            targets = [c for c in rest if viewsheds[c.object_id].contains_xy([(x, y)])[0]
                       and (due_at is None or t_check is None or due_at[c.object_id] <= t_check + 1e-9)]
            option = self._endpoint_option(kind, index, x, y, heading_in, heading_out, targets, t_check)
            if option is not None:
                options.append(option)
                taken = {c.object_id for c in option.candidates}
                rest = [c for c in rest if c.object_id not in taken]
        return options

    def _endpoint_option(self, kind, index, x, y, heading_in, heading_out, targets, t_check=None):
        targets = sorted(targets, key=lambda c: (-self.value(c, t_check), c.object_id))  # most valuable first
        while targets:
            ordered = sorted(targets, key=lambda c: wrap_angle(math.atan2(c.y - y, c.x - x) - heading_in))
            cost = endpoint_cost_s(x, y, heading_in, heading_out, ordered, self.turn_rate, self.dwell_s)
            if self.max_cost_s <= 0 or cost <= self.max_cost_s:
                stop = Stop(index, x, y, heading_out, [(c.object_id, c.x, c.y) for c in ordered])
                return ObservationOption(kind=kind, resume=REMAINING_PATH, stop=stop, candidates=ordered,
                                         values=[self.value(c, t_check) for c in ordered], cost_s=cost)
            targets.pop()  # drop the least valuable target
        return None

    # ---------- Detours ----------

    def detour_eligible(self, candidate, now, exclude=()):
        """Not excluded, out of cooldown, and with a known importance (no belief threshold)."""
        if candidate.object_id in exclude:
            return False
        last = self.last_checked.get(candidate.object_id)
        if last is not None and now - last < self.cooldown_s:
            return False
        return self.importance_m_fn is not None and self.importance_m_fn(candidate) is not None

    def checkable_from_path(self, path_xy, viewshed, start_xy=None):
        """Visible from a point an opportunistic or endpoint stop may use: a usable path point (outside
        the skip zones), the park pose path_xy[-1], or the start (x, y) if the robot may look from there."""
        path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
        if len(path_xy) == 0:
            return False
        points = [path_xy[usable_points(path_xy, self.skip_start_m, self.skip_goal_m)], path_xy[-1:]]
        if start_xy is not None:
            points.append(np.asarray(start_xy, dtype=float).reshape(1, 2))
        return bool(viewshed.contains_xy(np.concatenate(points)).any())

    def detour_candidates(self, path_xy, candidates, viewsheds, now, exclude=(), covered=None, start_xy=None,
                          times=None):
        """Eligible objects no opportunistic or endpoint stop serves: the only ones a detour may serve.

        Left out: objects in covered (they have a stop on this path), and objects not yet due an
        opportunistic check by the end of this path (belief still above check_below_belief then) that a
        usable point sees (checkable_from_path): they get a stop once their belief has decayed.
        covered=None computes it with opportunistic_options (start heading taken from the path).
        """
        path_xy = np.asarray(path_xy, dtype=float)
        tt = self._path_times(path_xy, times)
        t_end = now + (float(tt[-1]) if len(tt) else 0.0)
        if covered is None:
            start_pose = None
            if start_xy is not None:
                start_pose = (start_xy[0], start_xy[1], float(path_headings(path_xy)[0]))
            covered = covered_ids(self.opportunistic_options(path_xy, candidates, viewsheds, now, exclude, start_pose,
                                                             times=times))
        out = []
        for c in candidates:
            vs = viewsheds.get(c.object_id)
            if vs is None or c.object_id in covered or not self.detour_eligible(c, now, exclude):
                continue
            if self.t_due(c, now, exclude) > t_end + 1e-9 and self.checkable_from_path(path_xy, vs, start_xy):
                continue
            out.append(c)
        return out

    def detour_value_m(self, candidate, t=None):
        """V of a detour check at time t (belief then; now: candidate.belief)."""
        p = self.detour_params
        return detour_value_m(self.importance_m_fn(candidate), self.belief_at(candidate, t), p.n_trips,
                              p.conclusive_prob)

    def max_detour_m(self, candidate, t=None):
        """D*: the longest detour that could pay off for this object alone (<= 0: never). t: the latest
        time the check could happen (lowest belief, largest V), so that D* stays an upper bound."""
        d_star = self.detour_value_m(candidate, t) - self.cruise_speed * self.dwell_s
        return min(d_star, self.detour_params.hard_cap_m)

    def detour_bound_time(self, now, times=None):
        """Latest time a detour on this path could check (end of the path plus the longest detour), for
        max_detour_m; None without times (V as of now)."""
        if times is None or not len(times):
            return None
        return now + float(times[-1]) + self.detour_params.hard_cap_m / self.cruise_speed

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

    def detour_options(self, path_xy, candidates, viewsheds, now, exclude=(), detours=None, covered=None,
                       start_xy=None, times=None):
        """Detours worth driving. Returns (chosen ObservationOptions, all DetourEvaluations).

        One evaluation per object with a detour result: its best viewpoint, bundled with every
        other detour candidate visible from there (values add). Options with V - C > margin are
        taken best-first by net benefit, without sharing objects, up to max_detours_per_path.
        Blocked ones (detour_blocked: branching off at the start or goal) are evaluated but never taken.
        covered / start_xy: as for detour_candidates(). times: trajectory time per path point; V uses
        the belief when the robot would look (leave point time + leave point -> viewpoint).
        """
        detours = detours or {}
        path_xy = np.asarray(path_xy, dtype=float)
        tt = self._path_times(path_xy, times)
        cands = {c.object_id: c for c in self.detour_candidates(path_xy, candidates, viewsheds, now, exclude,
                                                                covered, start_xy, times)}
        evaluations = []
        for object_id, c in cands.items():
            d = detours.get(object_id)
            if d is None:
                continue
            vx, vy = d.viewpoint_xy
            to_view_s = d.to_view_m / self.cruise_speed
            t_look = now + float(tt[min(d.leave_index, len(tt) - 1)]) + to_view_s if times is not None else None
            eta = t_look if t_look is not None else now + to_view_s
            winner = self.peer_yield(DETOUR, c.object_id, eta)
            targets = [u for u in self._bundle(c, cands, viewsheds, vx, vy, d.arrival_heading)
                       if u is c or self.peer_yield(DETOUR, u.object_id, eta) is None]  # others check those first
            values = [self.detour_value_m(u, t_look) for u in targets]
            cost_m = self.detour_cost_m(d, targets)
            stop = Stop(d.leave_index, vx, vy, d.arrival_heading, [(u.object_id, u.x, u.y) for u in targets])
            option = ObservationOption(
                kind=DETOUR, resume=REPLAN, stop=stop, candidates=targets, values=values,
                cost_s=cost_m / self.cruise_speed, detour_m=d.detour_m, value_m=sum(values), cost_m=cost_m,
                leave_xy=tuple(d.leave_xy), to_view_s=to_view_s,
            )
            blocked = self.detour_blocked(d) or ("robot %d checks first" % winner.robot_id if winner else "")
            evaluations.append(ObservationEvaluation([u.object_id for u in targets], d.detour_m, sum(values),
                                                     cost_m, option=option, values_m=values,
                                                     blocked=blocked, source=d))
        p = self.detour_params
        chosen = self._choose(evaluations, p.max_detours_per_path, p.margin_m, self.max_cost_s)
        return chosen, evaluations

    # ---------- Scouts ----------

    def peer_yield(self, kind, object_id, eta):
        """The other robot's intent a check of object_id at eta gives way to (yields_to), or None."""
        if self.robot_id is None or not self.peer_intents:
            return None
        mine = [o for o in self.peer_intents if o.object_id == object_id]
        return yields_to(kind, eta, self.robot_id, mine, self.intent_margin_s) if mine else None

    def covered_elsewhere(self, candidate):
        """True if the object will be checked without a scout. Peer checks are compared per scout in
        scout_options (they depend on the scout's ETA); nothing else is known here."""
        return False

    def scout_eligible(self, candidate, now, exclude=()):
        """As detour_eligible, and not covered elsewhere (no belief threshold)."""
        return self.detour_eligible(candidate, now, exclude) and not self.covered_elsewhere(candidate)

    def scout_candidates(self, candidates, viewsheds, now, exclude=()):
        """Eligible objects with a viewshed."""
        return [c for c in candidates if c.object_id in viewsheds and self.scout_eligible(c, now, exclude)]

    def scout_value_m(self, candidate, t=None):
        """V of a scout check at time t (belief then; now: candidate.belief)."""
        p = self.scout_params
        return detour_value_m(self.importance_m_fn(candidate), self.belief_at(candidate, t), p.n_trips,
                              p.conclusive_prob)

    def max_scout_m(self, candidate, now=None):
        """Longest start -> viewpoint distance that could pay off for this object alone (<= 0: never).
        now: V is taken at now + hard_cap / v (the latest arrival searched), so the limit is an upper bound."""
        t = None if now is None else now + self.scout_params.hard_cap_m / self.cruise_speed
        limit = self.scout_cost.travel_limit_m(self.scout_value_m(candidate, t), self)
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
            to_view_s = r.to_view_m / self.cruise_speed
            eta = now + to_view_s
            winner = self.peer_yield(SCOUT, c.object_id, eta)
            targets = [u for u in self._bundle(c, cands, viewsheds, vx, vy, r.arrival_heading)
                       if u is c or self.peer_yield(SCOUT, u.object_id, eta) is None]  # others check those first
            values = [self.scout_value_m(u, eta) for u in targets]
            parts = self.scout_cost.breakdown(r, targets, self)
            cost_m = parts["total_m"]
            stop = Stop(-1, vx, vy, r.arrival_heading, [(u.object_id, u.x, u.y) for u in targets])
            option = ObservationOption(
                kind=SCOUT, resume=RETURN, stop=stop, candidates=targets, values=values,
                cost_s=cost_m / self.cruise_speed, detour_m=r.to_view_m, value_m=sum(values), cost_m=cost_m,
                leave_xy=tuple(r.start_xy), to_view_s=to_view_s,
            )
            evaluations.append(ObservationEvaluation([u.object_id for u in targets], r.to_view_m, sum(values),
                                                     cost_m, option=option, values_m=values, kind=SCOUT,
                                                     source=r, breakdown=parts,
                                                     blocked="robot %d checks first" % winner.robot_id if winner else ""))
        p = self.scout_params
        return self._choose(evaluations, p.max_scouts, p.margin_m), evaluations

    # ---------- Revalidation (when the check is about to happen) ----------

    def revalidate(self, kind, object_ids, candidates, now, planned_eta=None, cost_s=0.0):
        """Re-apply the rules to a planned check just before it happens. Returns [(due, reason)] per object.

        candidates: {object_id: Candidate}, the current ledger state (a missing object is gone). The
        rules are evaluated at max(now, planned_eta): arriving early must not cancel a check the plan
        timed for later; a new observation (own or a peer's), a removal or a finished check changes them.
          OPPORTUNISTIC / AT_START / AT_GOAL: eligible_at (in the ledger, no cooldown, belief <= threshold).
          DETOUR / SCOUT: in the ledger, no cooldown, and the remaining targets together still worth it:
                          V(t) - cost_s * cruise_speed > margin (cost_s: what is still to be spent).
        """
        t = now if planned_eta is None else max(now, planned_eta)
        out = []
        for oid in object_ids:
            c = candidates.get(oid)
            if c is None:
                out.append((False, "gone"))
            elif self.in_cooldown(c, t):
                out.append((False, "checked %.0f s ago" % (t - self.last_checked[oid])))
            elif kind not in (DETOUR, SCOUT) and self.belief_at(c, t) > self.check_below_belief:
                out.append((False, "belief %.2f > %.2f" % (self.belief_at(c, t), self.check_below_belief)))
            else:
                out.append((True, ""))
        if kind in (DETOUR, SCOUT) and self.importance_m_fn is not None:
            live = [candidates[oid] for oid, (due, _r) in zip(object_ids, out) if due]
            value_fn, p = (self.detour_value_m, self.detour_params) if kind == DETOUR else \
                (self.scout_value_m, self.scout_params)
            value = sum(value_fn(c, t) for c in live if self.importance_m_fn(c) is not None)
            cost = cost_s * self.cruise_speed
            if live and value - cost <= p.margin_m:
                reason = "not worth it now (V=%.1f m, C=%.1f m)" % (value, cost)
                out = [(False, reason) if due else (due, r) for due, r in out]
        return out

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
