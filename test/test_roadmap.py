"""Unit tests for navigation_utils/roadmap.py: python3 -m pytest mattbot_navigation/test"""

import logging
import math

import networkx as nx
import numpy as np

from navigation_utils.roadmap import RoadmapParams, load_or_build_roadmap
from synthetic_maps import RES, corridor, free, l_corridor, roadmap_of, t_junction, walls


def test_corridor_is_a_chain_of_spaced_nodes():
    rm = roadmap_of(corridor(width=1.6, length=10.0))
    g = rm.graph
    assert nx.is_connected(g)
    assert max(d for _n, d in g.degree()) <= 2
    assert sum(1 for _n, d in g.degree() if d == 1) == 2  # a path: two ends
    weights = [w for _u, _v, w in g.edges(data="weight")]
    assert all(0.75 <= w <= 2.25 for w in weights)  # ~corridor_node_spacing (1.5 m)
    # Skeleton spans the C-space corridor: 10 m minus ~robot radius at each end
    assert 8.0 <= sum(weights) <= 10.5
    assert np.allclose(rm.node_xy[:, 1], 2.0, atol=0.2)


def test_edge_weight_is_polyline_length_not_chord():
    rm = roadmap_of(l_corridor())
    g = rm.graph
    ends = [n for n, d in g.degree() if d == 1]
    assert len(ends) == 2
    a, b = ends
    path_len = nx.shortest_path_length(g, a, b, weight="weight")
    assert path_len > np.linalg.norm(rm.node_xy[a] - rm.node_xy[b]) + 1.0  # goes round the corner
    for u, v, d in g.edges(data=True):
        chord = np.linalg.norm(rm.node_xy[u] - rm.node_xy[v])
        assert d["weight"] >= chord - 1e-9
        poly = d["polyline"]
        steps = np.hypot(*np.diff(poly, axis=0).T).sum() * rm.res
        assert math.isclose(d["weight"], steps)


def test_t_junction_has_one_degree_three_node():
    rm = roadmap_of(t_junction())
    degrees = dict(rm.graph.degree())
    j = [n for n, d in degrees.items() if d == 3]
    assert len(j) == 1
    assert np.allclose(rm.node_xy[j[0]], (6.0, 1.8), atol=0.4)


def test_open_area_gets_lattice_nodes_with_free_edges():
    g = walls(12.0, 12.0)
    free(g, 1.0, 1.0, 11.0, 11.0)  # 10 m square room
    rm = roadmap_of(g)
    kinds = nx.get_node_attributes(rm.graph, "kind")
    assert sum(1 for k in kinds.values() if k == "open") >= 9
    for _u, _v, poly in rm.graph.edges(data="polyline"):
        assert rm.cfree[poly[:, 0], poly[:, 1]].all()


def test_narrow_area_gets_no_lattice():
    rm = roadmap_of(corridor(width=3.0))
    assert "open" not in set(nx.get_node_attributes(rm.graph, "kind").values())


def test_disconnected_rooms_keep_largest_and_warn(caplog):
    g = walls(16.0, 6.0)
    free(g, 1.0, 1.0, 9.0, 3.0)  # big corridor
    free(g, 11.0, 1.0, 14.0, 3.0)  # separate small one
    with caplog.at_level(logging.WARNING):
        rm = roadmap_of(g)
    assert nx.is_connected(rm.graph)
    assert (rm.node_xy[:, 0] < 9.5).all()
    assert any("disconnected component" in r.message for r in caplog.records)


def test_nearest_node_must_be_visible():
    g = walls(14.0, 7.0)
    free(g, 1.0, 1.0, 13.0, 4.0)  # corridor A, 3 m wide, centre y = 2.5
    free(g, 1.0, 4.1, 13.0, 5.7)  # corridor B, centre y = 4.9, behind a 0.1 m wall
    free(g, 1.0, 1.0, 2.6, 5.7)  # joined at the west end
    rm = roadmap_of(g)
    b_nodes = [n for n in rm.graph if rm.node_xy[n, 1] > 4.5 and rm.node_xy[n, 0] > 4.0]
    assert b_nodes
    bx, by = rm.node_xy[b_nodes[0]]
    q = (bx, 3.9)  # in A, 1.0 m from the B node through the wall
    nearest_euclid = int(np.argmin(np.hypot(*(rm.node_xy - q).T)))
    assert rm.node_xy[nearest_euclid, 1] > 4.0  # the trap: Euclidean nearest is in B
    n = rm.nearest_node(*q)
    assert rm.node_xy[n, 1] < 4.0


def test_edges_near_radius():
    rm = roadmap_of(corridor(width=1.6))
    far_cell = np.array([[int(round(2.0 / rm.res)) + 15, int(round(6.0 / rm.res))]])  # 3 m above the axis
    assert rm.edges_near(far_cell, 2.0) == set()
    assert rm.edges_near(far_cell, 3.5)


def test_cache_hit_and_rebuild_on_change(tmp_path):
    g = corridor()
    args = (g.shape[1], g.shape[0], RES, (0.0, 0.0))
    rm1, cached1 = load_or_build_roadmap(g.ravel(), *args, params=RoadmapParams(), cache_dir=str(tmp_path))
    rm2, cached2 = load_or_build_roadmap(g.ravel(), *args, params=RoadmapParams(), cache_dir=str(tmp_path))
    assert not cached1 and cached2
    assert rm2.fingerprint == rm1.fingerprint
    assert rm2.graph.number_of_edges() == rm1.graph.number_of_edges()

    g2 = g.copy()
    g2[0, 0] = 0  # one cell changed
    _rm, cached3 = load_or_build_roadmap(g2.ravel(), *args, params=RoadmapParams(), cache_dir=str(tmp_path))
    assert not cached3
    _rm, cached4 = load_or_build_roadmap(g.ravel(), *args, params=RoadmapParams(robot_radius=0.3),
                                         cache_dir=str(tmp_path))
    assert not cached4


def test_unknown_counts_as_occupied():
    g = corridor(width=1.6)
    free(g, 6.0, 1.0, 6.2, 3.0, value=-1)  # unknown band across the corridor
    rm = roadmap_of(g)
    assert (rm.node_xy[:, 0] < 6.0).all() or (rm.node_xy[:, 0] > 6.2).all()  # split; one side kept


def test_float32_resolution_from_ros_message_gives_same_roadmap():
    g = corridor()
    res32 = float(np.float32(RES))  # OccupancyGrid.info.resolution is float32
    assert res32 != RES
    a = roadmap_of(g)
    from navigation_utils.roadmap import build_roadmap
    b = build_roadmap(g.ravel(), g.shape[1], g.shape[0], res32, (np.float32(0.0), 0.0), RoadmapParams())
    assert a.fingerprint == b.fingerprint
    assert a.graph.number_of_nodes() == b.graph.number_of_nodes()
    assert np.array_equal(a.cfree, b.cfree)
