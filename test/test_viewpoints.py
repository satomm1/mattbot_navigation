"""Unit tests for navigation_utils/viewpoints.py: python3 -m pytest mattbot_navigation/test"""

import math

import numpy as np

from navigation_utils.viewpoints import blocking_mask, compute_viewshed, nearby_blockers, select_stops

RES = 0.1


def empty_map(w=100, h=60):
    return np.zeros((h, w), dtype=bool)


def cell(xy):
    return (int(math.floor(xy[1] / RES)), int(math.floor(xy[0] / RES)))  # (row, col)


def test_blocking_mask_occupied_and_unknown():
    m = blocking_mask([0, 100, -1, 49, 50, 0], 3, 2)
    assert m.tolist() == [[False, True, True], [False, True, False]]


def test_range_annulus():
    vs = compute_viewshed(empty_map(), (5.0, 3.0), (0.0, 0.0), RES, r_min=1.0, r_max=2.0)
    assert vs.contains_xy([(6.5, 3.0)])[0]  # 1.5 m
    assert not vs.contains_xy([(5.5, 3.0)])[0]  # 0.5 m: too close
    assert not vs.contains_xy([(7.5, 3.0)])[0]  # 2.5 m: too far
    assert not vs.contains_xy([(5.0, 3.0)])[0]  # object cell itself


def test_wall_blocks_line_of_sight():
    blocking = empty_map()
    blocking[:, 60] = True  # vertical wall at x = 6.0..6.1
    vs = compute_viewshed(blocking, (5.0, 3.0), (0.0, 0.0), RES, r_min=0.5, r_max=3.0)
    assert vs.contains_xy([(5.8, 3.0), (3.0, 3.0)]).all()  # in front of the wall / behind the object
    assert not vs.contains_xy([(7.0, 3.0)])[0]  # behind the wall
    assert not vs.contains_xy([(6.05, 3.0)])[0]  # the wall cell


def test_object_next_to_wall_still_visible():
    blocking = empty_map()
    blocking[30, 50] = True  # object drawn into the static map
    vs = compute_viewshed(blocking, (5.05, 3.05), (0.0, 0.0), RES, r_min=0.5, r_max=2.0)
    assert vs.contains_xy([(6.5, 3.05)])[0]


def test_map_edges_and_origin():
    vs = compute_viewshed(empty_map(), (-4.8, 0.3), (-5.0, 0.0), RES, r_min=0.5, r_max=2.0)
    assert vs.i0 == 0 and vs.j0 == 0  # clipped to the map
    assert vs.contains_xy([(-3.8, 0.3)])[0]
    assert not vs.contains_xy([(-6.0, 0.3)])[0]  # off map


def test_object_on_map_edge_with_exact_range():
    # Regression: a sample at exactly r_max rounded into the cell outside the bounding box
    vs = compute_viewshed(empty_map(), (8.0, 0.0), (0.0, 0.0), RES, r_min=1.0, r_max=3.5)
    assert vs.contains_xy([(8.0, 3.0)])[0]


def straight_path(x0=0.0, x1=10.0, y=1.0, n=101):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, y)])


def views(*objs, r_min=1.0, r_max=3.5, blocking=None):
    blocking = empty_map() if blocking is None else blocking
    return {oid: compute_viewshed(blocking, (x, y), (0.0, 0.0), RES, r_min, r_max) for oid, x, y in objs}


def test_stop_nearest_preferred_distance():
    path = straight_path()
    obj = ("a", 5.0, 3.0)  # 2 m off the path
    (stop,) = select_stops(path, [obj], views(obj), r_pref=2.0)
    assert stop.x == 5.0 and stop.targets == [obj]
    assert stop.heading == 0.0


def test_object_not_visible_from_path_gets_no_stop():
    blocking = empty_map()
    blocking[20, :] = True  # wall along y = 2.0 between path and object
    obj = ("a", 5.0, 3.0)
    assert select_stops(straight_path(), [obj], views(obj, blocking=blocking)) == []


def test_skips_start_and_goal_zones():
    path = straight_path(0.0, 3.0, n=31)
    obj = ("a", 1.5, 1.0 + 1e-3)  # on the path; viewshed covers path except within r_min
    (stop,) = select_stops(path, [obj], views(obj, r_min=0.2), r_pref=0.0, skip_start_m=1.0, skip_goal_m=1.0)
    assert 1.0 <= stop.x <= 2.0


def test_one_stop_per_object_and_path_order():
    path = straight_path()
    objs = [("b", 8.0, 3.0), ("a", 2.0, 3.0)]
    stops = select_stops(path, objs, views(*objs))
    assert [s.targets[0][0] for s in stops] == ["a", "b"]
    assert [s.x for s in stops] == [2.0, 8.0]


def test_nearby_stops_merge_into_one():
    path = straight_path(y=3.0)
    objs = [("a", 5.0, 5.0), ("b", 5.3, 1.0)]  # left and right of the path, best seen from x ~ 5.0-5.3
    (stop,) = select_stops(path, objs, views(*objs), merge_m=0.5)
    assert stop.x == 5.0
    # Sweep order by signed angle from the path heading: right (b, negative) before left (a)
    assert [t[0] for t in stop.targets] == ["b", "a"]


def test_unknown_viewshed_skipped():
    obj = ("a", 5.0, 3.0)
    assert select_stops(straight_path(), [obj], {}) == []


def test_known_object_blocks_line_of_sight():
    obj = (5.0, 3.0)
    vs = compute_viewshed(empty_map(), obj, (0.0, 0.0), RES, 0.5, 3.0, blockers=[(5.0, 2.0, 0.4)])
    assert not vs.contains_xy([(5.0, 1.0)])[0]  # straight behind the blocker
    assert vs.contains_xy([(5.0, 4.5), (6.5, 3.0), (3.5, 3.0)]).all()  # other directions
    far = compute_viewshed(empty_map(), obj, (0.0, 0.0), RES, 0.5, 3.0, blockers=[(9.0, 0.5, 0.4)])
    plain = compute_viewshed(empty_map(), obj, (0.0, 0.0), RES, 0.5, 3.0)
    assert np.array_equal(far.mask, plain.mask)  # blocker out of reach changes nothing


def test_no_stop_when_known_object_hides_target_from_path():
    path = straight_path(4.0, 6.0, y=1.0, n=21)  # path segment 3 m below the target
    target = ("a", 5.0, 4.0)
    box = [(5.0, 2.5, 2.0)]  # known object strictly between them (y 1.5..3.5, x 4..6)
    vs = {"a": compute_viewshed(empty_map(), (5.0, 4.0), (0.0, 0.0), RES, 1.0, 3.5, blockers=box)}
    assert vs["a"].contains_xy([(5.0, 5.5)])[0]  # still visible from the other side
    assert select_stops(path, [target], vs, skip_start_m=0.0, skip_goal_m=0.0) == []
    vs = {"a": compute_viewshed(empty_map(), (5.0, 4.0), (0.0, 0.0), RES, 1.0, 3.5)}
    assert len(select_stops(path, [target], vs, skip_start_m=0.0, skip_goal_m=0.0)) == 1


def test_nearby_blockers_signature():
    objs = [("a", 5.0, 3.0, 0.5), ("b", 5.0, 2.0, 0.4), ("c", 20.0, 20.0, 0.4)]
    blockers, sig = nearby_blockers("a", (5.0, 3.0), objs, 4.5)
    assert blockers == [(5.0, 2.0, 0.4)]  # excludes itself and far objects
    _, sig_moved = nearby_blockers("a", (5.0, 3.0), [("a", 5.0, 3.0, 0.5), ("b", 5.4, 2.0, 0.4)], 4.5)
    assert sig != sig_moved
    _, sig_same = nearby_blockers("a", (5.0, 3.0), [("a", 5.0, 3.0, 0.5), ("b", 5.02, 2.0, 0.4)], 4.5)
    assert sig == sig_same  # sub-0.1 m jitter does not invalidate the cache


def test_prefers_stop_before_object_over_overshoot():
    # 0.8 m off the path: r_pref is reached both before (small turn) and after the object (~157 deg).
    # The point after is slightly closer to r_pref, but the stop must not overshoot.
    obj = ("a", 5.0, 1.8)
    for path in (straight_path(), straight_path(10.0, 0.0)):
        vs = views(obj)
        far_side = 5.0 + math.sqrt(2.0 ** 2 - 0.8 ** 2) * np.sign(path[-1, 0] - path[0, 0])
        assert vs["a"].contains_xy([(far_side, 1.0)])[0]
        (stop,) = select_stops(path, [obj], vs, r_pref=2.0)
        bearing = math.atan2(obj[2] - stop.y, obj[1] - stop.x)
        assert abs(math.degrees(math.atan2(math.sin(bearing - stop.heading), math.cos(bearing - stop.heading)))) < 90
        # Without the preference both sides tie on distance
        (free,) = select_stops(path, [obj], vs, r_pref=2.0, max_turn=None)
        assert abs(math.hypot(obj[1] - free.x, obj[2] - free.y) - 2.0) <= abs(
            math.hypot(obj[1] - stop.x, obj[2] - stop.y) - 2.0)


def test_overshoot_stop_used_when_object_only_visible_behind():
    blocking = empty_map()
    blocking[11:25, 0:60] = True  # wall between the path and the object, up to x = 6.0
    obj = ("a", 6.3, 2.0)  # just past the wall end: seen only from x > 6.3 at >= r_min
    vs = views(obj, r_min=1.2, blocking=blocking)
    (stop,) = select_stops(straight_path(0.0, 8.0, n=81), [obj], vs, r_pref=2.0)
    assert stop.x > obj[1]  # past the object: it can only be seen looking back
