"""Unit tests for navigation_utils/observation_policy.py: python3 -m pytest mattbot_navigation/test"""

import math

import numpy as np
import pytest

from navigation_utils.observation_policy import (
    AT_GOAL,
    AT_START,
    OPPORTUNISTIC,
    REMAINING_PATH,
    Candidate,
    ThresholdPolicy,
    DETOUR,
    SCOUT,
    check_etas,
    yields_to,
    covered_ids,
    endpoint_cost_s,
    opportunistic_cost_s,
)
from navigation_utils.viewpoints import Stop, Viewshed, compute_viewshed

RES = 0.1


def cand(oid, belief, x=5.0, y=3.0):
    return Candidate(oid, "chair", x, y, belief)


def test_eligibility_threshold_exclude_cooldown():
    p = ThresholdPolicy(check_below_belief=0.5, cooldown_s=10.0)
    assert p.eligible(cand("a", 0.5), now=0.0)
    assert not p.eligible(cand("a", 0.6), now=0.0)
    assert not p.eligible(cand("a", 0.2), now=0.0, exclude={"a"})
    p.record_check("a", 100.0)
    assert not p.eligible(cand("a", 0.2), now=105.0)
    assert p.eligible(cand("a", 0.2), now=110.0)


def test_value_uses_belief():
    p = ThresholdPolicy()
    assert p.value(cand("a", 0.25)) == pytest.approx(0.75)


def test_value_uses_importance_fn():
    importance = {"a": 4.0, "b": 0.0}
    p = ThresholdPolicy(importance_fn=lambda c: importance[c.object_id])
    assert p.value(cand("a", 0.25)) == pytest.approx(3.0)
    assert p.value(cand("b", 0.25)) == 0.0
    # Importance does not change which objects are eligible
    assert p.eligible(cand("b", 0.25), now=0.0)


def test_cost_turn_out_dwell_and_back():
    # Path heading 0; one target straight left (+90 deg): turn 90 out, 90 back
    stop = Stop(0, 0.0, 0.0, 0.0, [("a", 0.0, 2.0)])
    assert opportunistic_cost_s(stop, turn_rate=1.0, dwell_s=3.0) == pytest.approx(math.pi + 3.0)
    # Two targets, right then left: 90 + 180 + 90 degrees of turning, two dwells
    stop = Stop(0, 0.0, 0.0, 0.0, [("b", 0.0, -2.0), ("a", 0.0, 2.0)])
    assert opportunistic_cost_s(stop, turn_rate=1.0, dwell_s=3.0) == pytest.approx(2 * math.pi + 6.0)


def setup_options(policy, candidates, now=0.0, exclude=()):
    path = np.column_stack([np.linspace(0, 10, 101), np.full(101, 1.0)])
    blocking = np.zeros((60, 100), dtype=bool)
    views = {c.object_id: compute_viewshed(blocking, (c.x, c.y), (0.0, 0.0), RES, 1.0, 3.5) for c in candidates}
    return policy.options(path, candidates, views, now, exclude)


def test_options_built_for_eligible_objects_only():
    cands = [cand("a", 0.2, x=2.0), cand("b", 0.9, x=8.0), cand("c", 0.4, x=8.0)]
    opts = setup_options(ThresholdPolicy(), cands, exclude={"a"})
    assert len(opts) == 1
    (opt,) = opts
    assert opt.kind == OPPORTUNISTIC and opt.resume == REMAINING_PATH
    assert [c.object_id for c in opt.candidates] == ["c"]
    assert opt.values == [pytest.approx(0.6)]
    assert opt.path_index == opt.stop.path_index and opt.cost_s > 0


def test_max_cost_drops_expensive_stops():
    cands = [cand("a", 0.2, x=2.0)]
    assert setup_options(ThresholdPolicy(max_cost_s=1.0), cands) == []
    assert len(setup_options(ThresholdPolicy(max_cost_s=0.0), cands)) == 1


# ---------- Endpoint checks and the detour hand-off (KNOWN_ISSUES 2.1 / 2.2) ----------
# Straight path (0, 1) -> (10, 1). Hand-made viewsheds: visible from the path only for x in the given ranges.

PATH = np.column_stack([np.linspace(0, 10, 101), np.full(101, 1.0)])
START = (0.0, 1.0, 0.0)  # robot at the path start, facing along it


def strip_view(obj_xy, *x_ranges):
    """Viewshed of an object at obj_xy that sees the path (y = 1) only for x in [lo, hi) of each range."""
    mask = np.zeros((60, 120), dtype=bool)
    for lo, hi in x_ranges:
        mask[10, int(round(lo / RES)):int(round(hi / RES))] = True
    return Viewshed(obj_xy, 0, 0, mask, (0.0, 0.0), RES)


def endpoint_policy(**kw):
    return ThresholdPolicy(check_below_belief=0.5, importance_m_fn=lambda c: 5.0, **kw)


def test_endpoint_cost_counts_only_extra_turning():
    # Facing 0, target straight left, leave facing 0: 90 out + 90 back, one dwell
    t = [cand("a", 0.0, x=0.0, y=2.0)]
    assert endpoint_cost_s(0.0, 0.0, 0.0, 0.0, t, turn_rate=1.0, dwell_s=3.0) == pytest.approx(math.pi + 3.0)
    # Leaving facing left anyway (e.g. the goal heading): looking costs no turning
    assert endpoint_cost_s(0.0, 0.0, 0.0, math.pi / 2, t, turn_rate=1.0, dwell_s=3.0) == pytest.approx(3.0)


def test_object_seen_only_at_the_start_gets_an_at_start_stop():
    c = cand("a", 0.0, x=0.2, y=3.0)
    views = {"a": strip_view((0.2, 3.0), (0.0, 0.4))}
    pol = endpoint_policy()
    (opt,) = pol.opportunistic_options(PATH, [c], views, now=0.0, start_pose=START)
    assert opt.kind == AT_START and opt.resume == REMAINING_PATH and opt.path_index == 0
    assert [u.object_id for u in opt.candidates] == ["a"] and opt.cost_s > 0
    assert pol.detour_candidates(PATH, [c], views, 0.0, covered=covered_ids([opt]), start_xy=START[:2]) == []


def test_without_a_start_check_it_is_left_to_detours():
    # Not the first plan for the goal: no AT_START, and no on-path stop may be placed in the skip zone
    c = cand("a", 0.0, x=0.2, y=3.0)
    views = {"a": strip_view((0.2, 3.0), (0.0, 0.4))}
    pol = endpoint_policy()
    assert pol.opportunistic_options(PATH, [c], views, now=0.0) == []
    assert [u.object_id for u in pol.detour_candidates(PATH, [c], views, 0.0, covered=frozenset())] == ["a"]
    assert [u.object_id for u in pol.detour_candidates(PATH, [c], views, 0.0)] == ["a"]  # covered computed


def test_object_seen_only_at_the_goal_gets_an_at_goal_stop():
    c = cand("a", 0.0, x=10.0, y=3.0)
    views = {"a": strip_view((10.0, 3.0), (9.7, 10.15))}
    (opt,) = endpoint_policy().opportunistic_options(PATH, [c], views, now=0.0, goal_theta=math.pi)
    assert opt.kind == AT_GOAL and opt.path_index == len(PATH) - 1
    assert (opt.stop.x, opt.stop.y, opt.stop.heading) == (10.0, 1.0, math.pi)


def test_start_is_preferred_and_on_path_stops_win():
    a = cand("a", 0.0, x=0.2, y=3.0)  # seen at both ends
    b = cand("b", 0.0, x=5.0, y=3.0)  # seen at both ends and mid-path
    views = {"a": strip_view((0.2, 3.0), (0.0, 0.4), (9.7, 10.15)),
             "b": strip_view((5.0, 3.0), (0.0, 0.4), (4.0, 6.0), (9.7, 10.15))}
    opts = endpoint_policy().opportunistic_options(PATH, [a, b], views, now=0.0, start_pose=START)
    assert [(o.kind, [u.object_id for u in o.candidates]) for o in opts] == [(OPPORTUNISTIC, ["b"]), (AT_START, ["a"])]


def test_high_belief_object_a_usable_point_sees_is_not_a_detour_candidate():
    # It gets an ordinary or endpoint stop once its belief decays
    pol = endpoint_policy()
    for oid, ranges in (("mid", [(4.0, 6.0)]), ("goal", [(9.7, 10.15)]), ("start", [(0.0, 0.4)])):
        c = cand(oid, 0.9, x=5.0, y=3.0)
        views = {oid: strip_view((5.0, 3.0), *ranges)}
        assert pol.opportunistic_options(PATH, [c], views, now=0.0, start_pose=START) == []
        assert pol.detour_candidates(PATH, [c], views, 0.0, covered=frozenset(), start_xy=START[:2]) == [], oid
    # Seen only inside the start skip zone, not from the start itself: nothing will ever stop there
    c = cand("zone", 0.9, x=0.3, y=3.0)
    views = {"zone": strip_view((0.3, 3.0), (0.1, 0.4))}
    assert [u.object_id for u in pol.detour_candidates(PATH, [c], views, 0.0, covered=frozenset(), start_xy=START[:2])] == ["zone"]


def test_stop_over_max_cost_hands_the_object_to_detours():
    # KNOWN_ISSUES 2.2: the on-path stop (90 deg out and back + dwell > 1 s) is dropped
    c = cand("a", 0.0, x=5.0, y=3.0)
    views = {"a": strip_view((5.0, 3.0), (4.0, 6.0))}
    pol = endpoint_policy(max_cost_s=1.0)
    opts = pol.opportunistic_options(PATH, [c], views, now=0.0, start_pose=START)
    assert opts == []
    assert [u.object_id for u in pol.detour_candidates(PATH, [c], views, 0.0, covered=covered_ids(opts), start_xy=START[:2])] == ["a"]


def test_endpoint_bundle_trimmed_by_value_under_max_cost():
    importance = {"hi": 4.0, "lo": 1.0}
    pol = ThresholdPolicy(max_cost_s=4.0, dwell_s=3.0, importance_fn=lambda c: importance[c.object_id])
    # Both straight ahead of the goal heading: no extra turning, 3 s each; only one fits in 4 s
    cs = [cand("lo", 0.0, x=12.0, y=1.05), cand("hi", 0.0, x=12.0, y=0.95)]
    views = {c.object_id: strip_view((c.x, c.y), (9.7, 10.15)) for c in cs}
    (opt,) = pol.opportunistic_options(PATH, cs, views, now=0.0, goal_theta=0.0)
    assert opt.kind == AT_GOAL and [u.object_id for u in opt.candidates] == ["hi"]
    assert opt.cost_s == pytest.approx(3.0, abs=0.1)


# ---------- Check timing (beliefs evaluated for when the check happens) ----------

def timed_policy(**kw):
    # hold 20 s, decay 40 s: an object seen at t = 0 has belief 1 until 20 s and 0.5 at 40 s
    return ThresholdPolicy(check_below_belief=0.5, hold_s=20.0, decay_s=40.0, importance_m_fn=lambda c: 5.0, **kw)


def seen(oid, at, x=5.0, y=3.0):
    return Candidate(oid, "chair", x, y, belief=1.0, last_stamp=at)


def test_belief_at_and_t_due():
    pol = timed_policy()
    c = seen("a", at=0.0)
    assert pol.belief_at(c, 10.0) == 1.0 and pol.belief_at(c, 40.0) == pytest.approx(0.5)
    assert pol.belief_at(c, None) == 1.0  # no time: the published belief
    assert pol.t_due(c, now=0.0) == pytest.approx(40.0)
    assert not pol.eligible_at(c, 39.0) and pol.eligible_at(c, 40.0)
    pol.record_check("a", 30.0)  # cooldown 120 s outlasts the decay
    assert pol.t_due(c, now=0.0) == pytest.approx(150.0)
    assert pol.t_due(c, now=0.0, exclude={"a"}) == math.inf
    # Without a belief model nothing is predicted: due now or never
    assert ThresholdPolicy().t_due(c, now=0.0) == math.inf
    assert ThresholdPolicy().t_due(cand("b", 0.3), now=0.0) == 0.0


def test_stop_planned_for_when_the_object_is_due():
    # Object at x = 5 seen at t = 0, due at 40 s; the robot passes x = 5 at 50 s (times = 10 s per metre)
    c = seen("a", at=0.0)
    views = {"a": strip_view((5.0, 3.0), (2.0, 8.0))}
    times = PATH[:, 0] * 10.0
    pol = timed_policy()
    assert pol.opportunistic_options(PATH, [c], views, now=0.0) == []  # not due now: nothing without times
    (opt,) = pol.opportunistic_options(PATH, [c], views, now=0.0, times=times)
    assert opt.kind == OPPORTUNISTIC and times[opt.path_index] >= 40.0  # only where it will be due
    assert opt.values == [pytest.approx(1.0 - pol.belief_at(c, times[opt.path_index]))]
    # Due only after the robot has passed every viewpoint: no stop on this path, and no detour either
    # (usable points see it, so a later path's stop will check it once it is due)
    late = seen("b", at=100.0)
    views_b = {"b": strip_view((5.0, 3.0), (2.0, 8.0))}
    assert pol.opportunistic_options(PATH, [late], views_b, now=0.0, times=times) == []
    assert pol.detour_candidates(PATH, [late], views_b, 0.0, covered=frozenset(), times=times) == []


def test_at_goal_only_if_due_on_arrival():
    c = seen("a", at=0.0, x=10.0)
    views = {"a": strip_view((10.0, 3.0), (9.7, 10.15))}
    pol = timed_policy()
    (opt,) = pol.opportunistic_options(PATH, [c], views, now=0.0, times=PATH[:, 0] * 5.0)  # arrive at 50 s
    assert opt.kind == AT_GOAL
    assert pol.opportunistic_options(PATH, [c], views, now=0.0, times=PATH[:, 0] * 3.0) == []  # arrive at 30 s


def test_check_etas_add_earlier_stops():
    class S:
        def __init__(self, kind, path_index, cost_s, to_view_s=0.0):
            self.kind, self.path_index, self.cost_s, self.to_view_s = kind, path_index, cost_s, to_view_s

    times = [0.0, 10.0, 20.0, 30.0]
    stops = [S(AT_GOAL, 3, 2.0), S(OPPORTUNISTIC, 2, 5.0), S(AT_START, 0, 4.0), S(DETOUR, 1, 7.0, to_view_s=3.0)]
    # Order: AT_START (0), DETOUR at 1 (10 + 4 + 3), OPPORTUNISTIC at 2 (20 + 4 + 7), AT_GOAL (30 + 4 + 7 + 5)
    assert check_etas(stops, times) == [pytest.approx(46.0), pytest.approx(31.0), 0.0, pytest.approx(17.0)]


def test_revalidate_uses_the_planned_time_and_new_information():
    pol = timed_policy()
    cands = {"a": seen("a", at=0.0)}
    # Planned for 45 s, robot early at 35 s: still due (evaluated at the planned time)
    assert pol.revalidate(OPPORTUNISTIC, ["a"], cands, now=35.0, planned_eta=45.0) == [(True, "")]
    # Re-sighted at 30 s: belief 1 again
    (due, reason), = pol.revalidate(OPPORTUNISTIC, ["a"], {"a": seen("a", at=30.0)}, now=45.0, planned_eta=45.0)
    assert not due and reason.startswith("belief 1.00")
    # Removed from the ledger
    assert pol.revalidate(AT_GOAL, ["a"], {}, now=45.0, planned_eta=45.0) == [(False, "gone")]
    # Checked meanwhile (cooldown)
    pol.record_check("a", 40.0)
    (due, reason), = pol.revalidate(AT_START, ["a"], cands, now=45.0, planned_eta=45.0)
    assert not due and reason.startswith("checked")


def test_revalidate_detour_needs_value_over_remaining_cost():
    pol = timed_policy()
    cands = {"a": seen("a", at=0.0)}
    # V = 5 trips * (1 - b)/2 * 5 m; at 60 s b = 0: V = 12.5 m
    assert pol.revalidate(DETOUR, ["a"], cands, now=60.0, planned_eta=60.0, cost_s=10.0) == [(True, "")]  # C = 4 m
    (due, reason), = pol.revalidate(DETOUR, ["a"], cands, now=60.0, planned_eta=60.0, cost_s=40.0)  # C = 16 m
    assert not due and reason.startswith("not worth it")
    # Re-sighted: V = 0, but no belief threshold for detours: only the value test drops it
    (due, reason), = pol.revalidate(DETOUR, ["a"], {"a": seen("a", at=55.0)}, now=60.0, planned_eta=60.0, cost_s=1.0)
    assert not due and reason.startswith("not worth it")


# ---------- Fleet coordination: yields_to ----------

class I:  # an intent (dds_utils.intents.Intent / mattbot_dds ObservationIntent shape)
    def __init__(self, robot_id, eta, kind=OPPORTUNISTIC, committed=False):
        self.robot_id, self.eta, self.kind, self.committed = robot_id, eta, kind, committed


def test_detour_yields_to_earlier_checks_with_margin_and_id_tiebreak():
    assert yields_to(DETOUR, 100.0, 5, [I(7, 85.0)]).robot_id == 7  # earlier by 15 > 10 s
    assert yields_to(DETOUR, 100.0, 5, [I(7, 95.0)]) is None  # within the margin, higher id
    assert yields_to(DETOUR, 100.0, 5, [I(3, 105.0)]).robot_id == 3  # within the margin, lower id
    assert yields_to(DETOUR, 100.0, 5, [I(3, 115.0)]) is None  # later by more than the margin
    assert yields_to(DETOUR, 100.0, 5, [I(7, 104.0, committed=True)]).robot_id == 7  # committed beats uncommitted
    assert yields_to(DETOUR, 100.0, 5, [I(5, 50.0)]) is None  # own intents never count
    assert yields_to(DETOUR, 100.0, 5, [I(7, 95.0), I(3, 104.0), I(9, 60.0)]).robot_id == 9  # earliest winner


def test_detour_commit_window():
    early = [I(7, 50.0)]
    assert yields_to(DETOUR, 100.0, 5, early, committed=True, remaining_s=20.0).robot_id == 7
    assert yields_to(DETOUR, 100.0, 5, early, committed=True, remaining_s=10.0) is None  # nearly there


def test_scout_yields_to_any_check_not_later_by_the_margin():
    assert yields_to(SCOUT, 100.0, 3, [I(7, 108.0)]).robot_id == 7  # later, but within the margin
    assert yields_to(SCOUT, 100.0, 3, [I(7, 115.0)]) is None
    assert yields_to(SCOUT, 100.0, 3, [I(7, 108.0, kind=SCOUT)]) is None  # scout vs scout: lower id keeps it
    assert yields_to(SCOUT, 100.0, 9, [I(7, 108.0, kind=SCOUT)]).robot_id == 7
    assert yields_to(SCOUT, 100.0, 3, [I(7, 0.0)], committed=True, remaining_s=1.0).robot_id == 7  # no commit window


def test_opportunistic_and_endpoint_never_yield():
    for kind in (OPPORTUNISTIC, AT_START, AT_GOAL):
        assert yields_to(kind, 100.0, 5, [I(1, 0.0, committed=True)]) is None


def test_planner_blocks_a_detour_another_robot_checks_first():
    class D:  # detour.DetourResult
        detour_m, viewpoint_xy, leave_index, leave_xy, to_view_m, arrival_heading = 2.0, (5.0, 1.0), 50, (5.0, 1.0), 0.4, 0.0
        branch_arc_m, path_length_m = 5.0, 10.0

    c = cand("a", 0.0, x=5.0, y=3.0)
    views = {"a": strip_view((5.0, 3.0))}  # hidden from the path
    pol = endpoint_policy(robot_id=5)
    chosen, (ev,) = pol.detour_options(PATH, [c], views, now=0.0, detours={"a": D()})
    assert chosen and not ev.blocked
    pol.peer_intents = [type("O", (), dict(object_id="a", robot_id=2, kind=OPPORTUNISTIC, eta=-20.0, committed=False))()]
    chosen, (ev,) = pol.detour_options(PATH, [c], views, now=0.0, detours={"a": D()})
    assert not chosen and ev.blocked == "robot 2 checks first"
