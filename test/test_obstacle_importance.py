"""Unit tests for navigation_utils/obstacle_importance.py: python3 -m pytest mattbot_navigation/test"""

import logging

import networkx as nx
import numpy as np
import pytest

from navigation_utils.obstacle_importance import (
    UNIFORM_LANDMARKS,
    WEIGHTED_LANDMARKS,
    ImportanceEvaluator,
    ImportanceParams,
    ObjectImportanceTracker,
    TripSet,
    build_trip_set,
)
from synthetic_maps import corridor, maze, ring, roadmap_of, t_junction


@pytest.fixture(scope="module")
def ring_rm():
    return roadmap_of(ring())


@pytest.fixture(scope="module")
def maze_rm():
    return roadmap_of(maze())


@pytest.fixture(scope="module")
def maze_eval(maze_rm):
    params = ImportanceParams(num_random_landmarks=20, seed=3)
    return ImportanceEvaluator(maze_rm, build_trip_set(maze_rm, params), params)


def write_landmarks(tmp_path, landmarks):
    path = tmp_path / "landmarks.yaml"
    path.write_text("".join("%s: [%r, %r]\n" % (n, x, y) for n, (x, y) in landmarks.items()))
    return str(path)


def brute_force_delta(rm, trips, B, d_max):
    """Per-trip extra distance with a full graph copy (independent of the evaluator)."""
    g = rm.graph.copy()
    g.remove_edges_from(B)
    out = []
    for s, t in zip(trips.landmark_nodes[trips.sources], trips.landmark_nodes[trips.targets]):
        d0 = nx.shortest_path_length(rm.graph, s, t, weight="weight")
        try:
            d1 = nx.shortest_path_length(g, s, t, weight="weight")
            out.append(min(d1 - d0, d_max))
        except nx.NetworkXNoPath:
            out.append(d_max)
    out = np.array(out)
    out[out < 1e-6] = 0.0
    return out


def test_ring_detour_matches_geometry(ring_rm):
    # Bottom corridor centre y = 1.8, top 6.2, sides x = 1.8 and 10.2
    trips = TripSet.from_pairs(ring_rm, [((3.0, 1.8), (9.0, 1.8))])
    ev = ImportanceEvaluator(ring_rm, trips)
    r = ev.evaluate_square("cart", 6.0, 1.8, 0.8)
    assert r.blocked_edges
    s, t = trips.landmark_nodes
    d_free = nx.shortest_path_length(ring_rm.graph, s, t, weight="weight")
    # Around the ring: 2 x 4.4 m up and down, plus the top corridor instead of the bottom one
    expected_detour = 2 * 4.4 + (10.2 - 1.8) - d_free + (ring_rm.node_xy[s, 0] - 1.8) + (10.2 - ring_rm.node_xy[t, 0])
    assert r.I_o == pytest.approx(expected_detour, rel=0.1)
    assert r.frac_affected == pytest.approx(1.0)
    assert r.num_affected == 1 and r.num_trips == 1


def test_dead_end_spur_costs_d_max(tmp_path):
    rm = roadmap_of(t_junction())  # corridor along y = 1.8, stem up x = 6 to y = 9
    lm = write_landmarks(tmp_path, {"west": (2.0, 1.8), "east": (10.0, 1.8), "spur": (6.0, 8.0)})
    params = ImportanceParams(landmarks_file=lm, d_max=50.0)
    trips = build_trip_set(rm, params)
    assert trips.landmark_names == ["west", "east", "spur"]
    ev = ImportanceEvaluator(rm, trips, params)
    r = ev.evaluate_square("box", 6.0, 5.0, 0.8)  # across the stem
    # 4 of the 6 ordered trips go to or from the spur and become impossible
    assert r.frac_affected == pytest.approx(4 / 6)
    assert r.I_o == pytest.approx(4 / 6 * 50.0)
    assert r.I_o_disc == pytest.approx(4 / 6 * 50.0)


def test_no_blocked_edges_returns_zero_without_search(maze_eval):
    runs = maze_eval.num_dijkstra_runs
    r = maze_eval.evaluate("far", coarse_cells=[[0, 0]])  # map corner, inside the wall
    assert r.blocked_edges == frozenset()
    assert (r.I_o, r.I_o_disc, r.frac_affected, r.mean_detour_if_affected, r.num_affected) == (0, 0, 0, 0, 0)
    assert maze_eval.num_dijkstra_runs == runs


def random_obstacles(rm, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        u, v = rm.edge_list[rng.integers(len(rm.edge_list))]
        poly = rm.graph.edges[u, v]["polyline"]
        r, c = poly[rng.integers(len(poly))]
        x, y = rm.cell_to_world(r, c)
        out.append(("o%d" % k, x, y, float(rng.choice([0.4, 0.6, 0.8]))))
    return out


def test_matches_brute_force_and_discovery_cost_dominates(maze_rm, maze_eval):
    p = maze_eval.params
    w = maze_eval.trips.weights
    n_blocking = 0
    for oid, x, y, width in random_obstacles(maze_rm, 25):
        r = maze_eval.evaluate_square(oid, x, y, width)
        delta = brute_force_delta(maze_rm, maze_eval.trips, r.blocked_edges, p.d_max)
        assert r.I_o == pytest.approx(float(np.dot(w, delta)), abs=1e-6)
        assert r.frac_affected == pytest.approx(float(w[delta > 0].sum()), abs=1e-9)
        assert r.num_affected == int((delta > 0).sum())
        assert r.I_o_disc >= r.I_o - 1e-9  # required: discovery is never cheaper
        assert r.I_o == pytest.approx(r.frac_affected * r.mean_detour_if_affected, abs=1e-9)
        n_blocking += bool(r.blocked_edges)
    assert n_blocking >= 10  # the check above was not vacuous


def test_weighted_trips(tmp_path, maze_rm, caplog):
    lm = write_landmarks(tmp_path, {"a": (2.0, 2.0), "b": (10.0, 2.0), "c": (10.0, 10.0), "d": (2.0, 10.0)})
    counts = tmp_path / "counts.csv"
    counts.write_text("start_landmark,goal_landmark,count\na,b,3\nb,a,1\na,b,2\nc,zz,5\nd,c,4\n")
    params = ImportanceParams(trip_mode=WEIGHTED_LANDMARKS, landmarks_file=lm, trip_counts_file=str(counts))
    with caplog.at_level(logging.WARNING):
        trips = build_trip_set(maze_rm, params)
    assert any("unknown landmark" in r.message for r in caplog.records)
    pairs = {(trips.landmark_names[s], trips.landmark_names[t]): w
             for s, t, w in zip(trips.sources, trips.targets, trips.weights)}
    assert pairs == pytest.approx({("a", "b"): 0.5, ("b", "a"): 0.1, ("d", "c"): 0.4})

    ev = ImportanceEvaluator(maze_rm, trips, params)
    r = ev.evaluate_square("x", 6.0, 2.0, 0.8)  # bottom corridor, between a and b
    delta = brute_force_delta(maze_rm, trips, r.blocked_edges, params.d_max)
    assert r.I_o == pytest.approx(float(np.dot(trips.weights, delta)))
    assert r.frac_affected == pytest.approx(0.6)  # a->b and b->a, not d->c


def test_weighted_mode_requires_named_landmarks(maze_rm):
    with pytest.raises(ValueError):
        build_trip_set(maze_rm, ImportanceParams(trip_mode=WEIGHTED_LANDMARKS))


def test_result_cache_by_blocked_edges(maze_rm, maze_eval):
    x, y = 6.0, 2.0
    r1 = maze_eval.evaluate_square("cached", x, y, 0.8)
    assert r1.blocked_edges
    runs = maze_eval.num_dijkstra_runs
    r2 = maze_eval.evaluate_square("cached", x + 0.02, y, 0.8)  # refined position, same edges
    assert r2 is r1
    assert maze_eval.num_dijkstra_runs == runs


def test_random_landmarks_are_deterministic(maze_rm):
    p = ImportanceParams(num_random_landmarks=10, seed=7)
    t1, t2 = build_trip_set(maze_rm, p), build_trip_set(maze_rm, p)
    assert np.array_equal(t1.landmark_nodes, t2.landmark_nodes)
    assert t1.key == t2.key
    assert len(t1) == 10 * 9 and t1.mode == UNIFORM_LANDMARKS
    assert t1.weights.sum() == pytest.approx(1.0)


def test_distance_table_cache(tmp_path, maze_rm):
    trips = build_trip_set(maze_rm, ImportanceParams(num_random_landmarks=8))
    a = ImportanceEvaluator(maze_rm, trips, cache_dir=str(tmp_path))
    assert list(tmp_path.glob("dist_*.npz"))
    b = ImportanceEvaluator(maze_rm, trips, cache_dir=str(tmp_path))
    assert np.array_equal(a.table.dist, b.table.dist)


# ---------- Compute once ----------


def test_other_obstacles_do_not_change_importance(maze_eval):
    tracker = ObjectImportanceTracker(maze_eval)
    tracker.update({"first": (6.0, 2.0, 0.8)})
    r_first = tracker.get("first")
    runs = maze_eval.num_dijkstra_runs

    # A second obstacle elsewhere: evaluated alone; the first one is not touched
    changed = tracker.update({"first": (6.0, 2.0, 0.8), "second": (6.0, 10.0, 0.8)})
    assert changed == {"second"}
    assert tracker.get("first") is r_first
    runs_after_second = maze_eval.num_dijkstra_runs

    # ... and it is the same as evaluating the first obstacle with nothing else around
    alone = ImportanceEvaluator(maze_eval.roadmap, maze_eval.trips, maze_eval.params, table=maze_eval.table)
    r_alone = alone.evaluate_square("first", 6.0, 2.0, 0.8)
    assert (r_alone.I_o, r_alone.I_o_disc, r_alone.blocked_edges) == (r_first.I_o, r_first.I_o_disc,
                                                                      r_first.blocked_edges)

    # Removing the second obstacle leaves the first unchanged, with no new searches
    assert tracker.update({"first": (6.0, 2.0, 0.8)}) == {"second"}
    assert tracker.get("first") is r_first
    assert maze_eval.num_dijkstra_runs == runs_after_second
    assert runs_after_second >= runs


def test_unchanged_footprint_is_never_recomputed(maze_eval):
    tracker = ObjectImportanceTracker(maze_eval)
    tracker.update({"a": (6.0, 2.0, 0.8)})
    r = tracker.get("a")
    runs = maze_eval.num_dijkstra_runs
    for _ in range(3):  # repeated belief updates with the same object
        assert tracker.update({"a": (6.0, 2.0, 0.8)}) == set()
    assert tracker.get("a") is r
    assert maze_eval.num_dijkstra_runs == runs


def test_footprint_change_recomputes_only_if_blocked_edges_change():
    rm = roadmap_of(corridor(width=3.0, length=12.0))  # hallway y in [0.5, 3.5]
    ev = ImportanceEvaluator(rm, TripSet.from_pairs(rm, [((2.0, 2.0), (12.0, 2.0))]))
    tracker = ObjectImportanceTracker(ev)
    tracker.update({"a": (7.0, 0.8, 0.5)})  # small cart against the wall: blocks nothing
    small = tracker.get("a")
    assert small.blocked_edges == frozenset() and small.I_o == 0.0

    # The ledger refines the position a little: footprint moves but still blocks nothing
    assert tracker.update({"a": (7.2, 0.8, 0.5)}) == set()

    # ... then sees it much wider: now it blocks the hallway and is recomputed
    assert tracker.update({"a": (7.2, 1.5, 2.0)}) == {"a"}
    big = tracker.get("a")
    assert big.blocked_edges and big.I_o == pytest.approx(ev.params.d_max)  # only route is cut
