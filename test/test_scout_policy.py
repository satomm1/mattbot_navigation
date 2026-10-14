"""Unit tests for the scout decision (observation_policy.py + scout.py)."""

import pytest

from navigation_utils.detour import DetourPlanner
from navigation_utils.observation_policy import (
    RETURN,
    SCOUT,
    Candidate,
    ScoutCostModel,
    ScoutValueParams,
    ThresholdPolicy,
    TravelTimeScoutCost,
    detour_value_m,
)
from navigation_utils.scout import ScoutPlanner, ScoutResult
from navigation_utils.viewpoints import compute_viewshed
from synthetic_maps import RES, corridor_with_room, roadmap_of

V_CRUISE, DWELL, TURN_RATE = 0.5, 4.0, 1.0
START = (2.0, 1.8)
BOX = (10.0, 7.0)  # in the side room, seen only through the door


@pytest.fixture(scope="module")
def room():
    g = corridor_with_room()
    scout = ScoutPlanner(DetourPlanner(roadmap_of(g)))
    blocking = (g >= 50) | (g < 0)
    return scout, blocking


def view(blocking, x, y):
    return compute_viewshed(blocking, (x, y), (0.0, 0.0), RES, 1.0, 3.5)


def policy(importance, cost=None, **params):
    """importance: {object_id: I_o or None}"""
    return ThresholdPolicy(
        cruise_speed=V_CRUISE, dwell_s=DWELL, turn_rate=TURN_RATE,
        importance_m_fn=lambda c: importance.get(c.object_id),
        scout_params=ScoutValueParams(**params), scout_cost=cost,
    )


def run(room, objects, importance, beliefs, cost=None, now=0.0, exclude=(), checked=None, **params):
    """objects {oid: (x, y)} -> (policy, chosen, evaluations, scout results)"""
    scout, blocking = room
    views = {oid: view(blocking, x, y) for oid, (x, y) in objects.items()}
    cands = [Candidate(oid, "cart", x, y, beliefs[oid]) for oid, (x, y) in objects.items()]
    pol = policy(importance, cost, **params)
    for oid, t in (checked or {}).items():
        pol.record_check(oid, t)
    sc = pol.scout_candidates(cands, views, now, exclude)
    targets = {c.object_id: (c.x, c.y, scout.viewpoint_nodes(views[c.object_id]), pol.max_scout_m(c)) for c in sc}
    results = scout.compute(START, targets)
    chosen, evals = pol.scout_options(START, cands, views, now, exclude, results)
    return pol, chosen, evals, results


def test_important_unknown_object_gets_a_scout(room):
    pol, chosen, evals, results = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0})
    assert len(chosen) == 1 and chosen[0].kind == SCOUT and chosen[0].resume == RETURN
    ev = evals[0]
    assert ev.chosen and ev.kind == SCOUT and ev.object_ids == ["box"]
    assert ev.value_m == pytest.approx(detour_value_m(20.0, 0.0, 5.0))  # 5 * 0.5 * 20 = 50 m
    r = results["box"]
    assert ev.detour_m == pytest.approx(r.to_view_m)
    # Placeholder cost: out and back + dwell (one target; turn >= 0)
    assert ev.cost_m >= 2 * r.to_view_m + V_CRUISE * DWELL - 1e-9
    assert chosen[0].stop.x == r.viewpoint_xy[0] and chosen[0].leave_xy == r.start_xy


def test_no_scout_when_just_seen_or_unimportant(room):
    for beliefs, importance in (({"box": 1.0}, {"box": 20.0}), ({"box": 0.0}, {"box": 0.0})):
        _pol, chosen, evals, _r = run(room, {"box": BOX}, importance, beliefs)
        assert chosen == [] and evals == []  # V = 0: never searched


def test_unevaluated_importance_gets_no_scout(room):
    _pol, chosen, evals, _r = run(room, {"box": BOX}, {"box": None}, {"box": 0.0})
    assert chosen == [] and evals == []


def test_excluded_and_cooldown_objects_get_no_scout(room):
    _p, chosen, _e, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, exclude={"box"})
    assert chosen == []
    _p, chosen, _e, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, now=100.0, checked={"box": 50.0})
    assert chosen == []
    _p, chosen, _e, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, now=500.0, checked={"box": 50.0})
    assert len(chosen) == 1


def test_break_even_and_margin(room):
    _p, _c, evals, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0})
    cost = evals[0].cost_m
    i_even = cost / (5.0 * 0.5)  # V = N p_gone I_o = C
    _p, chosen, _e, _r = run(room, {"box": BOX}, {"box": i_even * 1.05}, {"box": 0.0})
    assert len(chosen) == 1
    _p, chosen, _e, _r = run(room, {"box": BOX}, {"box": i_even * 0.95}, {"box": 0.0})
    assert chosen == []
    _p, chosen, evals, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, margin_m=evals[0].net_m + 1.0)
    assert chosen == [] and len(evals) == 1


def test_bundled_objects_add_value_and_max_scouts(room):
    objects = {"a": BOX, "b": (9.0, 6.5)}
    _p, chosen, evals, _r = run(room, objects, {"a": 20.0, "b": 20.0}, {"a": 0.0, "b": 0.0})
    assert len(chosen) == 1  # max_scouts = 1, and the bundles share objects
    assert sorted(chosen[0].stop.targets[k][0] for k in range(2)) == ["a", "b"]
    assert chosen[0].value_m == pytest.approx(2 * detour_value_m(20.0, 0.0, 5.0))


def test_cost_model_is_pluggable(room):
    class Prohibitive(ScoutCostModel):
        name = "prohibitive"

        def cost_m(self, result, targets, policy):
            return 1e6

    _p, chosen, evals, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, cost=Prohibitive())
    assert chosen == [] and evals[0].cost_m == 1e6


def test_one_way_cost_drops_the_return_leg(room):
    _p, _c, out_back, results = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0})
    _p, _c, one_way, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0},
                              cost=TravelTimeScoutCost(round_trip=False))
    assert out_back[0].cost_m - one_way[0].cost_m == pytest.approx(results["box"].return_m)


def test_search_limit_follows_the_cost_model(room):
    c = Candidate("box", "cart", *BOX, 0.0)
    budget = detour_value_m(8.0, 0.0, 5.0) - V_CRUISE * DWELL  # 20 - 2 = 18 m, below the 30 m cap
    assert policy({"box": 8.0}).max_scout_m(c) == pytest.approx(budget / 2)
    assert policy({"box": 8.0}, TravelTimeScoutCost(False)).max_scout_m(c) == pytest.approx(budget)
    assert policy({"box": 8.0}, hard_cap_m=3.0).max_scout_m(c) == 3.0


# ---------- Opportunity cost (TravelTimeScoutCost fixed_s, per_s_away) ----------


def fake_result(to_view_m, heading=0.0, view=(0.0, 0.0)):
    return ScoutResult("x", view, (0.0, 0.0), to_view_m, to_view_m, heading)


def test_defaults_are_the_physical_cost(room):
    _p, _c, plain, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0})
    _p, _c, zero, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0},
                           cost=TravelTimeScoutCost(round_trip=True, fixed_s=0.0, per_s_away=0.0))
    assert zero[0].cost_m == pytest.approx(plain[0].cost_m)


def test_fixed_cost_adds_v_times_fixed_s_and_can_flip_the_decision(room):
    _p, chosen, base, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0})
    assert len(chosen) == 1
    net = base[0].net_m
    _p, chosen, ev, _r = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0},
                             cost=TravelTimeScoutCost(fixed_s=10.0))
    assert ev[0].cost_m - base[0].cost_m == pytest.approx(V_CRUISE * 10.0)
    assert len(chosen) == 1
    over = 1.01 * net / V_CRUISE  # just more than the net benefit
    pol, chosen, ev, results = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0},
                                   cost=TravelTimeScoutCost(fixed_s=over))
    assert chosen == []
    # Pruned before the search: even the cheapest scout to its viewpoint costs more than V
    assert ev == [] and results == {}
    assert pol.max_scout_m(Candidate("box", "cart", *BOX, 0.0)) < base[0].detour_m


def test_time_away_and_its_cost(room):
    pol, _c, ev, results = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0},
                               cost=TravelTimeScoutCost(per_s_away=0.5))
    r = results["box"]
    plain = TravelTimeScoutCost()
    b0 = plain.breakdown(r, ev[0].option.candidates, pol)
    b = pol.scout_cost.breakdown(r, ev[0].option.candidates, pol)
    look_s = b0["look_m"] / V_CRUISE  # turn / turn_rate + dwell * n
    assert look_s >= DWELL
    assert b["t_away_s"] == pytest.approx((r.to_view_m + r.return_m) / V_CRUISE + look_s)
    assert b["away_m"] == pytest.approx(V_CRUISE * 0.5 * b["t_away_s"])
    assert ev[0].cost_m == pytest.approx(b0["total_m"] + b["away_m"])
    assert b["total_m"] == pytest.approx(b["travel_m"] + b["look_m"] + b["fixed_m"] + b["away_m"])
    assert ev[0].breakdown == pytest.approx(b)  # stored on the evaluation for the message and the log


def test_one_way_time_away_drops_the_return_leg():
    pol = policy({})
    r = fake_result(6.0)
    there_and_back = TravelTimeScoutCost(per_s_away=1.0).breakdown(r, [], pol)["t_away_s"]
    one_way = TravelTimeScoutCost(round_trip=False, per_s_away=1.0).breakdown(r, [], pol)["t_away_s"]
    assert there_and_back - one_way == pytest.approx(6.0 / V_CRUISE)


def test_time_away_widens_the_gap_between_near_and_far_scouts():
    pol = policy({})
    target = [Candidate("t", "cart", 1.0, 0.0, 0.0)]  # straight ahead: no turn
    near, far = fake_result(2.0), fake_result(8.0)
    gap0 = TravelTimeScoutCost().cost_m(far, target, pol) - TravelTimeScoutCost().cost_m(near, target, pol)
    lam = 0.5
    c = TravelTimeScoutCost(fixed_s=30.0, per_s_away=lam)
    gap = c.cost_m(far, target, pol) - c.cost_m(near, target, pol)
    assert gap0 == pytest.approx(2 * 6.0)
    assert gap == pytest.approx(gap0 * (1 + lam))  # fixed_s cancels; time away scales the difference


@pytest.mark.parametrize("fixed_s,lam,round_trip", [(0.0, 0.0, True), (30.0, 0.0, True), (0.0, 0.5, True),
                                                     (20.0, 1.0, False), (200.0, 0.0, True)])
def test_search_limit_is_the_break_even_distance(fixed_s, lam, round_trip):
    cost = TravelTimeScoutCost(round_trip=round_trip, fixed_s=fixed_s, per_s_away=lam)
    pol = policy({"box": 20.0}, cost)
    c = Candidate("box", "cart", 1.0, 0.0, 0.0)
    value = pol.scout_value_m(c)  # 50 m
    k = 2.0 if round_trip else 1.0
    expected = ((value - V_CRUISE * fixed_s) / (1 + lam) - V_CRUISE * DWELL) / k
    assert pol.max_scout_m(c) == pytest.approx(min(expected, 30.0))
    if 0 < expected < 30.0:
        # One target straight ahead of the arrival heading: V - C = 0 exactly at the limit
        assert cost.cost_m(fake_result(expected), [c], pol) == pytest.approx(value)
    if fixed_s == 200.0:
        assert pol.max_scout_m(c) <= 0  # fixed cost alone exceeds V: never searched


def test_custom_model_without_breakdown_still_works(room):
    class Flat(ScoutCostModel):
        name = "flat"

        def cost_m(self, result, targets, policy):
            return 5.0

    pol, chosen, ev, results = run(room, {"box": BOX}, {"box": 20.0}, {"box": 0.0}, cost=Flat())
    assert len(chosen) == 1
    assert Flat().breakdown(results["box"], ev[0].option.candidates, pol) == {"total_m": 5.0}
    assert pol.max_scout_m(Candidate("box", "cart", *BOX, 0.0)) == pytest.approx(min(50.0 - V_CRUISE * DWELL, 30.0))
