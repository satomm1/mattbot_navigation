"""Unit tests for navigation_utils/scout.py: python3 -m pytest mattbot_navigation/test"""

import math

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from navigation_utils.detour import DetourPlanner
from navigation_utils.scout import ScoutParams, ScoutPlanner
from navigation_utils.viewpoints import compute_viewshed
from synthetic_maps import RES, corridor_with_room, maze, roadmap_of


def setup(grid):
    rm = roadmap_of(grid)
    blocking = (grid >= 50) | (grid < 0)
    detour = DetourPlanner(rm)
    return rm, detour, ScoutPlanner(detour), blocking


def target(scout, blocking, x, y, r_min=1.0, r_max=3.5):
    vs = compute_viewshed(blocking, (x, y), (0.0, 0.0), RES, r_min, r_max)
    return (x, y, scout.viewpoint_nodes(vs))


@pytest.fixture(scope="module")
def room():
    return setup(corridor_with_room())


START = (2.0, 1.8)  # west end of the corridor


def brute_force(detour, start_xy, nodes):
    s = int(detour.snap_nodes([start_xy], 0.5)[0])
    d = dijkstra(detour.graph, directed=True, indices=s)
    return float(np.min(d[nodes]))


def test_object_in_side_room_is_seen_through_the_door(room):
    _rm, detour, scout, blocking = room
    t = target(scout, blocking, 10.0, 7.0)
    r = scout.compute(START, {"box": t})["box"]
    # Along the corridor to the door, then into it until the object is within r_max
    assert 6.0 <= r.to_view_m <= 11.0
    assert r.to_view_m == pytest.approx(brute_force(detour, START, t[2]), abs=1e-6)
    assert r.return_m == r.to_view_m
    assert r.viewpoint_xy[1] > 2.6  # in the door or the room
    vx, vy = r.viewpoint_xy
    assert 1.0 - RES * 2 <= math.hypot(vx - 10.0, vy - 7.0) <= 3.5 + 0.2 * math.sqrt(2)
    np.testing.assert_allclose(r.route_xy[-1], r.viewpoint_xy)
    assert np.hypot(*(r.route_xy[0] - np.array(START))) < 0.3


def test_object_visible_from_the_start_needs_no_driving(room):
    _rm, _detour, scout, blocking = room
    r = scout.compute(START, {"cart": target(scout, blocking, 4.0, 2.2)}, start_heading=0.7)["cart"]
    assert r.to_view_m == pytest.approx(0.0)
    assert r.arrival_heading == pytest.approx(0.7)  # already there: keeps the robot's heading


def test_matches_brute_force_in_a_maze():
    rm, detour, scout, blocking = setup(maze())
    start = (2.0, 2.0)
    rng = np.random.default_rng(1)
    free_xy = detour.xy[rng.choice(detour.n, size=40, replace=False)]
    # Short sight range so that most objects need real driving
    targets = {"o%d" % k: target(scout, blocking, x, y, r_min=0.3, r_max=1.5) for k, (x, y) in enumerate(free_xy)}
    targets = {oid: t for oid, t in targets.items() if len(t[2])}
    out = scout.compute(start, targets, ScoutParams(max_scout_m=1e6, r_max=1.5))
    assert set(out) == set(targets)  # nothing wrongly skipped by the lower bound
    assert sum(r.to_view_m > 3.0 for r in out.values()) >= 10
    for oid, r in out.items():
        assert r.to_view_m == pytest.approx(brute_force(detour, start, targets[oid][2]), abs=1e-6)
        slack = rm.res / math.sqrt(2.0)
        assert ScoutPlanner.lower_bound(r.start_xy, targets[oid][:2], 1.5, slack) <= r.to_view_m + 1e-9


def test_far_objects_skipped_without_a_search(room):
    _rm, _detour, scout, blocking = room
    before = scout.num_searches
    out = scout.compute(START, {"box": target(scout, blocking, 10.0, 7.0) + (2.0,)})
    assert out == {} and scout.num_searches == before  # lower bound > 2 m: no Dijkstra


def test_per_object_limit_drops_further_results(room):
    _rm, _detour, scout, blocking = room
    t = target(scout, blocking, 10.0, 7.0)
    exact = scout.compute(START, {"box": t})["box"].to_view_m
    assert scout.compute(START, {"box": t + (exact + 0.1,)})
    # Passes the lower bound (~4 m) but the viewpoint is further
    assert scout.compute(START, {"box": t + (exact - 0.5,)}) == {}


def test_limit_zero_or_negative_is_never_searched(room):
    _rm, _detour, scout, blocking = room
    before = scout.num_searches
    t = target(scout, blocking, 4.0, 2.2)
    assert scout.compute(START, {"a": t + (0.0,), "b": t + (-1.0,)}) == {}
    assert scout.num_searches == before


def test_start_outside_cspace_gives_nothing(room):
    _rm, _detour, scout, blocking = room
    assert scout.compute((0.1, 0.1), {"cart": target(scout, blocking, 4.0, 2.2)}) == {}


def test_viewpoints_are_cspace_cells(room):
    rm, detour, scout, blocking = room
    r = scout.compute(START, {"box": target(scout, blocking, 10.0, 7.0)})["box"]
    k = int(detour.snap_nodes([r.viewpoint_xy], 0.0)[0])
    assert k >= 0
    row, col = detour.cells[k]
    assert rm.cfree[row, col]
