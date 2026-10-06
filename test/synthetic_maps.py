"""Small synthetic occupancy grids (ROS layout, 0.05 m cells, origin (0, 0)) for roadmap tests."""

import numpy as np

from navigation_utils.roadmap import RoadmapParams, build_roadmap

RES = 0.05


def walls(w_m, h_m):
    """All-occupied map of w_m x h_m metres."""
    return np.full((int(round(h_m / RES)), int(round(w_m / RES))), 100, dtype=np.int8)


def free(grid, x0, y0, x1, y1, value=0):
    """Set the rectangle [x0, x1) x [y0, y1) (metres) to value (0 free, 100 occupied, -1 unknown)."""
    grid[int(round(y0 / RES)):int(round(y1 / RES)), int(round(x0 / RES)):int(round(x1 / RES))] = value
    return grid


def roadmap_of(grid, **params):
    return build_roadmap(grid.ravel(), grid.shape[1], grid.shape[0], RES, (0.0, 0.0), RoadmapParams(**params))


def corridor(width=1.6, length=10.0):
    """Straight horizontal corridor x in [1, 1 + length), centred on y = 2."""
    g = walls(length + 2.0, 4.0 + max(0.0, width - 2.0))
    return free(g, 1.0, 2.0 - width / 2, 1.0 + length, 2.0 + width / 2)


def l_corridor():
    g = walls(12.0, 12.0)
    free(g, 1.0, 1.0, 11.0, 2.6)
    return free(g, 9.4, 1.0, 11.0, 11.0)


def t_junction():
    g = walls(12.0, 10.0)
    free(g, 1.0, 1.0, 11.0, 2.6)
    return free(g, 5.2, 1.0, 6.8, 9.0)


def ring():
    """Two 1.6 m corridors (y centres 1.8 and 6.2) joined at both ends (x centres 1.8, 10.2)."""
    g = walls(12.0, 8.0)
    free(g, 1.0, 1.0, 11.0, 7.0)
    return free(g, 2.6, 2.6, 9.4, 5.4, value=100)


def maze():
    """3 x 3 grid of 1.6 m corridors (centres x, y in {2, 6, 10}) plus two dead-end spurs."""
    g = walls(13.0, 13.0)
    for c in (2.0, 6.0, 10.0):
        free(g, 1.2, c - 0.8, 10.8, c + 0.8)
        free(g, c - 0.8, 1.2, c + 0.8, 10.8)
    free(g, 10.8, 5.2, 12.6, 6.8)  # spur east from (10, 6)
    free(g, 3.0, 9.2, 4.0, 10.8, value=100)  # wall in the top corridor between x = 2 and 6
    return g


def corridor_with_room():
    """Corridor y in [1, 2.6], x in [1, 15]; room x in [7, 11], y in [4, 8] above it, joined by a
    1.2 m door (x in [8.4, 9.6]) - the room interior is out of sight from the corridor beyond ~3 m."""
    g = walls(16.0, 9.0)
    free(g, 1.0, 1.0, 15.0, 2.6)
    free(g, 7.0, 4.0, 11.0, 8.0)
    return free(g, 8.4, 2.6, 9.6, 4.0)
