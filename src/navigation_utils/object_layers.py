"""Occupancy layers drawn from ledger objects (pure numpy, no ROS).

Navigation routes around every ledger object until it is removed (seen gone): objects with belief
b >= 0 (just seen = 1, decayed to unknown = 0) are blocked; removed objects (b = -1) are not in
the ledger at all. Footprints are squares of the object's width centred on its local-map position,
drawn with the same inclusive floor cell range as mattbot_dds dds_utils/belief.belief_grid.
"""

import math

import numpy as np

BLOCKED = 100
FREE = -1  # "no information" in /object_map (the navigator only treats > 85 as occupied)
MIN_SIDE_M = 0.2  # smallest footprint side (same as viewpoints.MIN_BLOCKER_M)


def footprint_cell_range(x, y, width, resolution, origin, grid_width, grid_height, min_side_m=MIN_SIDE_M):
    """Inclusive (i0, i1, j0, j1) column / row range of a square footprint, clipped to the grid,
    or None if it lies entirely off the grid. Always contains the centre cell."""
    half = max(float(width), min_side_m) / 2.0
    i0 = int(math.floor((x - half - origin[0]) / resolution))
    i1 = int(math.floor((x + half - origin[0]) / resolution))
    j0 = int(math.floor((y - half - origin[1]) / resolution))
    j1 = int(math.floor((y + half - origin[1]) / resolution))
    i0, i1 = max(i0, 0), min(i1, grid_width - 1)
    j0, j1 = max(j0, 0), min(j1, grid_height - 1)
    if i0 > i1 or j0 > j1:
        return None
    return i0, i1, j0, j1


def ledger_blockout_grid(objects, width, height, resolution, origin, min_side_m=MIN_SIDE_M):
    """(height, width) int8 grid: BLOCKED under every object with belief >= 0, else FREE.

    objects: iterable of (x, y, width, belief) in the local map frame.
    """
    grid = np.full((height, width), FREE, dtype=np.int8)
    for x, y, w, belief in objects:
        if belief < 0:
            continue
        r = footprint_cell_range(x, y, w, resolution, origin, width, height, min_side_m)
        if r is not None:
            i0, i1, j0, j1 = r
            grid[j0:j1 + 1, i0:i1 + 1] = BLOCKED
    return grid
