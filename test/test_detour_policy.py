"""Unit tests for the value-of-information detour rule (observation_policy.py)."""

import math

import numpy as np
import pytest

from navigation_utils.detour import DetourPlanner
from navigation_utils.observation_policy import (
    DETOUR,
    OPPORTUNISTIC,
    REPLAN,
    Candidate,
    DetourValueParams,
    ThresholdPolicy,
    detour_value_m,
    p_gone,
)
from navigation_utils.viewpoints import compute_viewshed
from synthetic_maps import RES, corridor_with_room, roadmap_of

V_CRUISE, DWELL, TURN_RATE = 0.5, 4.0, 1.0


@pytest.fixture(scope="module")
def room():
    g = corridor_with_room()
    rm = roadmap_of(g)
    planner = DetourPlanner(rm)
    blocking = (g >= 50) | (g < 0)
    path = planner.shortest_path_xy((2.0, 1.8), (14.0, 1.8))
    return planner, blocking, path


def view(blocking, x, y):
    return compute_viewshed(blocking, (x, y), (0.0, 0.0), RES, 1.0, 3.5)


def policy(importance, **params):
    """importance: {object_id: I_o or None}"""
    return ThresholdPolicy(
        check_below_belief=0.5, cruise_speed=V_CRUISE, dwell_s=DWELL, turn_rate=TURN_RATE,
        importance_m_fn=lambda c: importance.get(c.object_id), detour_params=DetourValueParams(**params),
    )


def run(room, objects, importance, beliefs, **params):
    """objects {oid: (x, y)} -> (policy, opportunistic options, chosen detours, evaluations, detours)"""
    planner, blocking, path = room
    views = {oid: view(blocking, x, y) for oid, (x, y) in objects.items()}
    cands = [Candidate(oid, "cart", x, y, beliefs[oid]) for oid, (x, y) in objects.items()]
    pol = policy(importance, **params)
    dc = pol.detour_candidates(path, cands, views, now=0.0)
    targets = {c.object_id: (c.x, c.y, planner.viewpoint_nodes(views[c.object_id]), pol.max_detour_m(c)) for c in dc}
    detours = planner.compute(path, targets)
    chosen, evals = pol.detour_options(path, cands, views, now=0.0, detours=detours)
    return pol, pol.opportunistic_options(path, cands, views, now=0.0), chosen, evals, detours, dc


def test_formulas():
    assert p_gone(1.0) == 0.0 and p_gone(0.0) == 0.5 and p_gone(0.4) == pytest.approx(0.3)
    assert p_gone(1.7) == 0.0 and p_gone(-0.2) == 0.5  # clamped to [0, 1]
    assert detour_value_m(6.8, 0.0, 10, 1.0) == pytest.approx(34.0)
    assert detour_value_m(6.8, 0.6, 10, 0.5) == pytest.approx(6.8)
    assert detour_value_m(0.0, 0.0, 10) == 0.0


def test_room_object_detour_value_and_cost(room):
    pol, _opp, chosen, evals, detours, _dc = run(room, {"box": (10.0, 7.0)}, {"box": 6.0}, {"box": 0.0}, n_trips=5)
    (ev,) = evals
    d = detours["box"]
    assert ev.value_m == pytest.approx(5 * 0.5 * 6.0)  # q N p_gone I_o = 15 m
    # C = D + v * (turn + dwell); the turn is between 0 and pi rad
    assert d.detour_m + V_CRUISE * DWELL <= ev.cost_m <= d.detour_m + V_CRUISE * (DWELL + math.pi)
    (opt,) = chosen
    assert opt.kind == DETOUR and opt.resume == REPLAN
    assert opt.path_index == d.leave_index  # DETOUR path_index = where to leave the path
    assert (opt.stop.x, opt.stop.y) == d.viewpoint_xy and opt.leave_xy == d.leave_xy
    assert opt.cost_s == pytest.approx(opt.cost_m / V_CRUISE)
    assert opt.values == [pytest.approx(15.0)]


def test_no_detour_when_just_seen_or_unimportant(room):
    for imp, b in [(6.0, 1.0), (0.0, 0.0)]:
        _pol, _o, chosen, evals, _d, dc = run(room, {"box": (10.0, 7.0)}, {"box": imp}, {"box": b})
        assert chosen == []
        assert evals == []  # D* <= 0: never even searched


def test_unevaluated_importance_gets_no_detour(room):
    _pol, _o, chosen, evals, _d, dc = run(room, {"box": (10.0, 7.0)}, {"box": None}, {"box": 0.0})
    assert dc == [] and chosen == [] and evals == []


def test_break_even_belief(room):
    """Going from 'skip' to 'go' happens where V = C, i.e. p_gone = C / (q N I_o)."""
    planner, blocking, path = room
    imp, n = 4.0, 5
    _pol, _o, _c, evals, _d, _dc = run(room, {"box": (10.0, 7.0)}, {"box": imp}, {"box": 0.0}, n_trips=n)
    cost = evals[0].cost_m
    b_star = 1.0 - 2.0 * cost / (n * imp)
    assert 0.0 < b_star < 1.0
    for b, go in [(b_star - 0.05, True), (b_star + 0.05, False)]:
        _pol, _o, chosen, _e, _d, _dc = run(room, {"box": (10.0, 7.0)}, {"box": imp}, {"box": b}, n_trips=n)
        assert bool(chosen) == go


def test_margin(room):
    _pol, _o, _c, evals, _d, _dc = run(room, {"box": (10.0, 7.0)}, {"box": 6.0}, {"box": 0.0})
    net = evals[0].net_m
    assert run(room, {"box": (10.0, 7.0)}, {"box": 6.0}, {"box": 0.0}, margin_m=net - 0.1)[2]
    assert not run(room, {"box": (10.0, 7.0)}, {"box": 6.0}, {"box": 0.0}, margin_m=net + 0.1)[2]


def test_objects_visible_from_path_are_left_to_opportunistic_stops(room):
    objects = {"hall": (5.0, 2.2), "near_start": (2.0, 2.2), "box": (10.0, 7.0)}
    beliefs = {"hall": 0.0, "near_start": 0.0, "box": 0.0}
    imp = {"hall": 50.0, "near_start": 50.0, "box": 6.0}
    _pol, opp, chosen, evals, detours, dc = run(room, objects, imp, beliefs)
    assert {c.object_id for c in dc} == {"box"}  # visible ones never reach the detour search
    assert set(detours) == {"box"}
    assert {oid for ev in evals for oid in ev.object_ids} == {"box"}  # and are never bundled
    assert "hall" in {c.object_id for o in opp for c in o.candidates}
    assert all(o.kind == DETOUR and o.candidates[0].object_id == "box" for o in chosen)


def test_visible_object_above_belief_threshold_still_not_a_detour(room):
    # b = 0.8 > check_below_belief: no opportunistic stop now, but it is visible from the path
    _pol, opp, chosen, evals, _d, dc = run(room, {"hall": (5.0, 2.2)}, {"hall": 50.0}, {"hall": 0.8})
    assert opp == [] and dc == [] and chosen == []


def test_excluded_and_cooldown_objects_get_no_detour(room):
    planner, blocking, path = room
    vs = {"box": view(blocking, 10.0, 7.0)}
    cands = [Candidate("box", "cart", 10.0, 7.0, 0.0)]
    pol = policy({"box": 6.0})
    assert pol.detour_candidates(path, cands, vs, now=0.0, exclude={"box"}) == []
    pol.record_check("box", 100.0)
    assert pol.detour_candidates(path, cands, vs, now=150.0) == []  # cooldown 120 s
    assert pol.detour_candidates(path, cands, vs, now=250.0)


def test_bundled_objects_add_value_and_one_detour_per_path(room):
    objects = {"box": (10.0, 7.0), "crate": (9.4, 6.4)}  # both seen from the doorway
    imp = {"box": 3.0, "crate": 3.0}
    beliefs = {"box": 0.0, "crate": 0.0}
    _pol, _o, chosen, evals, _d, _dc = run(room, objects, imp, beliefs)
    assert any(set(ev.object_ids) == {"box", "crate"} for ev in evals)
    best = max(evals, key=lambda e: e.net_m)
    assert best.value_m == pytest.approx(2 * detour_value_m(3.0, 0.0, 5))
    assert len(chosen) == 1 and {t[0] for t in chosen[0].stop.targets} == {"box", "crate"}
    # Alone, one of them would not be worth it; together they are
    alone = run(room, {"box": (10.0, 7.0)}, {"box": 3.0}, {"box": 0.0})
    assert alone[3][0].net_m < best.net_m


def test_best_net_benefit_wins(room):
    objects = {"box": (10.0, 7.0)}
    _pol, _o, chosen, evals, _d, _dc = run(room, objects, {"box": 6.0}, {"box": 0.0}, max_detours_per_path=1)
    assert len(chosen) <= 1
    assert all(ev.chosen == (ev.option in chosen) for ev in evals)


def test_opportunistic_unchanged_without_detours(room):
    planner, blocking, path = room
    vs = {"hall": view(blocking, 5.0, 2.2)}
    cands = [Candidate("hall", "cart", 5.0, 2.2, 0.2)]
    pol = policy({"hall": 3.0})
    opts = pol.options(path, cands, vs, now=0.0)
    assert [o.kind for o in opts] == [OPPORTUNISTIC]
    assert opts[0].path_index == opts[0].stop.path_index
