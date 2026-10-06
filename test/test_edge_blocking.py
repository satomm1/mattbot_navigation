"""Unit tests for navigation_utils/edge_blocking.py: python3 -m pytest mattbot_navigation/test"""

import numpy as np

from navigation_utils.edge_blocking import blocked_edges, footprint_cells
from synthetic_maps import corridor, roadmap_of, t_junction


def blocked(rm, **footprint):
    return blocked_edges(rm, footprint_cells(rm, **footprint))


def test_cart_against_wall_of_wide_hallway_blocks_nothing():
    """Required case: a partial blockage that leaves room for the robot must not block."""
    rm = roadmap_of(corridor(width=3.0))  # hallway y in [0.5, 3.5]
    assert blocked(rm, square=(6.0, 0.75, 0.5)) == frozenset()  # 0.5 m cart touching the bottom wall
    assert blocked(rm, square=(6.0, 3.25, 0.5)) == frozenset()  # ... and the top wall


def test_cart_in_middle_of_wide_hallway_blocks_nothing():
    rm = roadmap_of(corridor(width=3.0))
    # 1.25 m free each side of the cart; the robot needs 0.8 m
    assert blocked(rm, square=(6.0, 2.0, 0.5)) == frozenset()


def test_obstacle_across_narrow_corridor_blocks_its_edge():
    rm = roadmap_of(corridor(width=1.2))
    b = blocked(rm, square=(6.0, 2.0, 0.5))
    assert len(b) >= 1
    for u, v in b:  # the blocked edge spans x = 6
        xs = sorted((rm.node_xy[u, 0], rm.node_xy[v, 0]))
        assert xs[0] - 0.5 <= 6.0 <= xs[1] + 0.5


def test_far_obstacle_has_no_candidates():
    rm = roadmap_of(corridor(width=1.6))
    # 2.5 m above the corridor axis (inside the wall region; still a valid footprint)
    assert blocked(rm, coarse_cells=[[22, 30]]) == frozenset()


def test_obstacle_on_junction_blocks_incident_edges():
    rm = roadmap_of(t_junction())
    j = next(n for n, d in rm.graph.degree() if d == 3)
    x, y = rm.node_xy[j]
    # 0.8 m cart in the middle of a 1.6 m T: no way past for a 0.8 m robot
    b = blocked(rm, square=(x, y, 0.8))
    incident = {tuple(sorted(e)) for e in rm.graph.edges(j)}
    assert incident <= b


def test_polygon_and_square_footprints_agree():
    rm = roadmap_of(corridor(width=1.2))
    x, y, w = 6.0, 2.0, 0.5
    poly = np.array([[x - w / 2, y - w / 2], [x + w / 2, y - w / 2], [x + w / 2, y + w / 2], [x - w / 2, y + w / 2]])
    assert blocked(rm, polygon=poly) == blocked(rm, square=(x, y, w))


def test_fine_cells_footprint():
    rm = roadmap_of(corridor(width=1.2))
    # fine (0.05 m) cells of a 0.5 m square at (6, 2)
    rr, cc = np.mgrid[35:45, 115:125]
    fine = np.stack([rr.ravel(), cc.ravel()], axis=1)
    assert blocked(rm, fine_cells=fine) == blocked(rm, square=(6.0, 2.0, 0.5))


def test_small_cart_on_junction_leaves_diagonal_gap():
    # A 0.4 m cart leaves ~0.85 m between its corners and the wall corners: the robot gets by
    rm = roadmap_of(t_junction())
    j = next(n for n, d in rm.graph.degree() if d == 3)
    incident = {tuple(sorted(e)) for e in rm.graph.edges(j)}
    assert len(incident - blocked(rm, square=(*rm.node_xy[j], 0.4))) >= 2


def test_obstacle_blocking_only_the_stem_of_a_t():
    rm = roadmap_of(t_junction())
    b = blocked(rm, square=(6.0, 5.0, 0.8))  # across the 1.6 m stem, away from the junction
    assert b
    for u, v in b:  # only stem edges (x ~ 6, y > 2.6)
        assert rm.node_xy[u, 1] > 2.0 or rm.node_xy[v, 1] > 2.0
