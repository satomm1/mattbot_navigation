"""Unit tests for navigation_utils/detour.py: python3 -m pytest mattbot_navigation/test"""

import math

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from navigation_utils.detour import DetourParams, DetourPlanner
from navigation_utils.viewpoints import compute_viewshed
from synthetic_maps import RES, corridor_with_room, maze, roadmap_of


def setup(grid):
    rm = roadmap_of(grid)
    blocking = (grid >= 50) | (grid < 0)
    return rm, DetourPlanner(rm), blocking


def target(planner, blocking, oid, x, y, r_min=1.0, r_max=3.5):
    vs = compute_viewshed(blocking, (x, y), (0.0, 0.0), RES, r_min, r_max)
    return vs, (x, y, planner.viewpoint_nodes(vs))


@pytest.fixture(scope="module")
def room():
    return setup(corridor_with_room())


def corridor_path(planner):
    return planner.shortest_path_xy((2.0, 1.8), (14.0, 1.8))


def test_object_in_side_room_needs_detour_through_door(room):
    rm, planner, blocking = room
    path = corridor_path(planner)
    _vs, t = target(planner, blocking, "box", 10.0, 7.0)
    d = planner.compute(path, {"box": t})["box"]
    assert not d.on_path
    # Into the doorway until the object is within r_max, then back out
    assert 1.5 <= d.detour_m <= 7.5
    assert d.to_view_m == pytest.approx(d.detour_m / 2, abs=1.0)
    assert 8.0 <= d.leave_xy[0] <= 10.0  # leaves the corridor at the door, not at the start
    assert d.viewpoint_xy[1] > 2.6  # the viewpoint is in the door or the room
    vx, vy = d.viewpoint_xy
    # A coarse cell is a viewpoint if any fine viewshed cell lies in it
    assert 1.0 - rm.res <= math.hypot(vx - 10.0, vy - 7.0) <= 3.5 + rm.res * math.sqrt(2)


def test_object_visible_from_path_is_on_path(room):
    _rm, planner, blocking = room
    path = corridor_path(planner)
    _vs, t = target(planner, blocking, "cart", 5.0, 2.2)
    d = planner.compute(path, {"cart": t})["cart"]
    assert d.on_path and d.detour_m == 0.0


def test_viewpoints_are_reachable_cspace_cells(room):
    rm, planner, blocking = room
    _vs, (_x, _y, nodes) = target(planner, blocking, "box", 10.0, 7.0)
    assert len(nodes)
    cells = planner.cells[nodes]
    assert rm.cfree[cells[:, 0], cells[:, 1]].all()


def brute_force(planner, path_xy, nodes):
    """min_i (arc_i + d(P_i, v)) + d(v, g) - L, with one Dijkstra per path node."""
    pn, arc = planner._path_nodes(path_xy, 0.5)
    first = {}
    for i, nd in enumerate(pn):
        if nd >= 0 and nd not in first:
            first[int(nd)] = arc[i]
    D = dijkstra(planner.graph, directed=True, indices=list(first))
    F = np.min(np.array(list(first.values()))[:, None] + D, axis=0)
    G = dijkstra(planner.graph, directed=True, indices=int(pn[-1]))
    L = F[int(pn[-1])]
    return float(np.min(F[nodes] + G[nodes] - L))


def test_matches_brute_force_and_far_objects_are_not_wrongly_skipped():
    rm, planner, blocking = setup(maze())
    path = planner.shortest_path_xy((2.0, 2.0), (10.0, 10.0))
    rng = np.random.default_rng(0)
    free_xy = planner.xy[rng.choice(planner.n, 80, replace=False)]
    targets = {}
    for k, (x, y) in enumerate(free_xy):  # short sight range so that many objects need detours
        _vs, targets["o%d" % k] = target(planner, blocking, "o%d" % k, x, y, r_min=0.3, r_max=1.5)
    everything = planner.compute(path, targets, DetourParams(max_detour_m=1e6, r_max=1.5))
    limited = planner.compute(path, targets, DetourParams(max_detour_m=8.0, r_max=1.5))
    checked = 0
    for oid, (_x, _y, nodes) in targets.items():
        if not len(nodes):
            continue
        exact = brute_force(planner, path, nodes)
        d = everything[oid]
        assert d.detour_m == pytest.approx(exact if exact > 0.3 else 0.0, abs=1e-6)
        if exact <= 8.0:
            assert oid in limited  # the far-object lower bound never drops a reachable object
        else:
            assert oid not in limited
        checked += 1
    assert checked >= 60
    assert sum(1 for d in everything.values() if d.detour_m > 0) >= 10


def test_lower_bound_is_below_exact():
    rm, planner, blocking = setup(maze())
    path = planner.shortest_path_xy((2.0, 2.0), (10.0, 2.0))
    _pn, arc = planner._path_nodes(path, 0.5)
    for x, y in [(2.0, 10.0), (10.0, 10.0), (6.0, 6.0), (12.0, 6.0)]:
        _vs, t = target(planner, blocking, "o", x, y)
        d = planner.compute(path, {"o": t}, DetourParams(max_detour_m=1e6))
        if "o" in d:
            assert planner.lower_bound(path, arc, (x, y), 3.5) <= d["o"].detour_m + 1e-6


def test_far_objects_skipped(room):
    _rm, planner, blocking = room
    path = planner.shortest_path_xy((2.0, 1.8), (6.0, 1.8))  # short path at the west end
    _vs, t = target(planner, blocking, "box", 10.0, 7.0)
    assert planner.compute(path, {"box": t}, DetourParams(max_detour_m=2.0)) == {}
    assert "box" in planner.compute(path, {"box": t}, DetourParams(max_detour_m=30.0))


# ---------- Per-object limits ----------


def test_object_limit_zero_or_negative_is_never_searched(room):
    _rm, planner, blocking = room
    path = corridor_path(planner)
    _vs, (x, y, nodes) = target(planner, blocking, "box", 10.0, 7.0)
    runs = planner.num_searches
    assert planner.compute(path, {"box": (x, y, nodes, 0.0)}) == {}
    assert planner.compute(path, {"box": (x, y, nodes, -3.0)}) == {}
    assert planner.num_searches == runs  # dropped before any Dijkstra


def test_per_object_limits_match_brute_force():
    rm, planner, blocking = setup(maze())
    path = planner.shortest_path_xy((2.0, 2.0), (10.0, 10.0))
    rng = np.random.default_rng(1)
    targets, exact = {}, {}
    for k, (x, y) in enumerate(planner.xy[rng.choice(planner.n, 60, replace=False)]):
        _vs, (x, y, nodes) = target(planner, blocking, "o%d" % k, x, y, r_min=0.3, r_max=1.5)
        if len(nodes):
            limit = float(rng.uniform(0.5, 10.0))
            targets["o%d" % k] = (x, y, nodes, limit)
            exact["o%d" % k] = brute_force(planner, path, nodes)
    got = planner.compute(path, targets, DetourParams(r_max=1.5))
    for oid, (_x, _y, _n, limit) in targets.items():
        e = exact[oid] if exact[oid] > 0.3 else 0.0
        if e <= limit:
            assert got[oid].detour_m == pytest.approx(e, abs=1e-6)
        else:
            assert oid not in got
