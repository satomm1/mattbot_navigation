"""StochOccupancyGrid2D.is_free: the precomputed footprint lookup gives exactly the window check's answers.
python3 -m pytest mattbot_navigation/test/test_grids.py"""

import copy

import numpy as np
import pytest

from navigation_utils.grids import StochOccupancyGrid2D
from navigation_utils import lattice_astar
from navigation_utils.search import AStar


def window_is_free(grid, state):
    """is_free before the lookup (the per-call footprint window), kept as the reference."""
    x, y = grid.snap_to_grid(state)
    grid_x = int((x - grid.origin_x) / grid.resolution)
    grid_y = int((y - grid.origin_y) / grid.resolution)
    half_config_size = int(round(grid.robot_d / 2 / grid.resolution))
    config_x_lower = max(0, grid_x - half_config_size)
    config_y_lower = max(0, grid_y - half_config_size)
    config_x_upper = min(grid.width, grid_x + half_config_size)
    config_y_upper = min(grid.height, grid_y + half_config_size)
    window = grid.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]
    if window.size == 0:
        return False
    return not np.any(window > 85)


def random_map(rng, h, w, fill=0.04, blob=6):
    """Walls round the edge, random rectangular blobs, some unknown (-1) and some below-threshold (50) cells."""
    g = np.zeros((h, w), dtype=np.int16)
    g[:2, :] = g[-2:, :] = g[:, :2] = g[:, -2:] = 100
    for _ in range(int(fill * h * w / blob)):
        r, c = rng.integers(0, h), rng.integers(0, w)
        g[r:r + rng.integers(1, blob), c:c + rng.integers(1, blob)] = rng.choice([100, 100, 90, 86, 85, 50, -1])
    return g


def make_grid(probs, resolution=0.05, origin=(0.0, 0.0), robot_d=0.6, lookup_after=None):
    h, w = probs.shape
    g = StochOccupancyGrid2D(resolution, w, h, origin[0], origin[1], 5, probs.ravel(), robot_d=robot_d)
    if lookup_after is not None:
        g.FOOTPRINT_LOOKUP_AFTER = lookup_after
    return g


@pytest.mark.parametrize("resolution,origin,robot_d", [
    (0.05, (0.0, 0.0), 0.6),
    (0.05, (-99.6, -62.8), 0.8),   # nonzero origin, even half size
    (0.05, (12.35, -3.1), 0.45),   # origin off the snap lattice, odd half size
    (0.1, (-5.0, 2.0), 0.6),
    (0.05, (0.0, 0.0), 0.0),       # empty window: never free
    (0.05, (0.0, 0.0), 0.05),      # half size rounds to 0 or 1
])
def test_lookup_matches_window_on_random_states(resolution, origin, robot_d):
    rng = np.random.default_rng(1)
    probs = random_map(rng, 120, 170)
    g = make_grid(probs, resolution, origin, robot_d, lookup_after=1)
    w_m, h_m = g.width * resolution, g.height * resolution
    xs = origin[0] + rng.uniform(-1.0, w_m + 1.0, 20000)  # includes states off the map
    ys = origin[1] + rng.uniform(-1.0, h_m + 1.0, 20000)
    for x, y in zip(xs, ys):
        assert g.is_free((x, y)) == window_is_free(g, (x, y)), (x, y)
    assert g._footprint_free is not None  # the lookup was used, not the fallback


def test_lookup_matches_window_on_every_cell():
    rng = np.random.default_rng(2)
    for robot_d in (0.3, 0.6, 1.0):
        g = make_grid(random_map(rng, 60, 80), robot_d=robot_d, origin=(-1.3, 4.45))
        free = g.footprint_free_grid()
        for row in range(g.height):
            for col in range(g.width):
                state = (g.origin_x + col * g.resolution, g.origin_y + row * g.resolution)
                gx = int((g.snap_to_grid(state)[0] - g.origin_x) / g.resolution)
                gy = int((g.snap_to_grid(state)[1] - g.origin_y) / g.resolution)
                if 0 <= gx < g.width and 0 <= gy < g.height:
                    assert free[gy, gx] == window_is_free(g, state), (robot_d, row, col)


def test_lookup_built_only_after_many_checks():
    g = make_grid(random_map(np.random.default_rng(3), 50, 50))
    for i in range(g.FOOTPRINT_LOOKUP_AFTER - 1):
        g.is_free((1.0 + 0.01 * (i % 50), 1.2))
    assert g._footprint_free is None
    g.is_free((1.0, 1.2))
    assert g._footprint_free is not None


def test_new_probs_or_robot_d_rebuilds_the_lookup():
    rng = np.random.default_rng(4)
    g = make_grid(random_map(rng, 80, 80), lookup_after=1)
    states = [tuple(s) for s in rng.uniform(0.0, 4.0, (3000, 2))]
    assert [g.is_free(s) for s in states] == [window_is_free(g, s) for s in states]

    combined = copy.copy(g)  # as the navigator's _planning_occupancy and the fast sim's Planner.grid do
    layer = np.zeros_like(g.probs)
    layer[20:60, 20:60] = 100
    combined.probs = np.maximum(g.probs, layer)
    assert [combined.is_free(s) for s in states] == [window_is_free(combined, s) for s in states]
    assert not combined.is_free((2.0, 2.0)) and combined._footprint_free[3] is not g._footprint_free[3]

    check = copy.copy(g)  # as path_still_valid does
    check.robot_d = 0.2
    assert [check.is_free(s) for s in states] == [window_is_free(check, s) for s in states]

    g.update(layer)
    assert [g.is_free(s) for s in states] == [window_is_free(g, s) for s in states]


class WindowGrid(object):
    """The grid as AStar saw it before the lookup: no footprint_lookup, so solve() takes its generic path."""

    def __init__(self, grid):
        self.grid = grid

    def is_free(self, state):
        return window_is_free(self.grid, state)


def plan(occupancy, extent, resolution, start, goal, robots=None, compiled=True):
    kw = {}
    if robots is not None:
        kw = dict(robots_x=[r[0] for r in robots], robots_y=[r[1] for r in robots], obj_x=[2.0], obj_y=[3.0],
                  obj_d=[0.5])
    problem = AStar((extent[0], extent[2]), (extent[1], extent[3]), start, goal, occupancy, resolution,
                    max_plan_time_sec=60.0, **kw)
    problem.use_compiled = compiled
    ok = problem.solve()
    return (problem.path if ok else None), problem


COMPILED = [False] + ([True] if lattice_astar.AVAILABLE else [])


@pytest.mark.parametrize("compiled", COMPILED)
@pytest.mark.parametrize("origin,grid_res,plan_res,with_robots", [
    ((-3.2, 1.75), 0.05, 0.05, False),
    ((0.0, 0.0), 0.05, 0.05, True),
    ((12.35, -3.1), 0.05, 0.1, False),  # planner lattice coarser than the grid: is_free re-snaps
    ((-3.2, 1.75), 0.1, 0.05, True),
])
def test_astar_paths_unchanged(origin, grid_res, plan_res, with_robots, compiled):
    rng = np.random.default_rng(5)
    probs = random_map(rng, 160, 240, fill=0.06, blob=12)
    grid = make_grid(probs, grid_res, origin, robot_d=0.6)
    ext = grid.extent
    cells = [(origin[0] + c * grid_res, origin[1] + r * grid_res)
             for r, c in zip(rng.integers(0, 160, 400), rng.integers(0, 240, 400))]
    cells = [(plan_res * round(x / plan_res), plan_res * round(y / plan_res)) for x, y in cells]
    cells = [s for s in cells if window_is_free(grid, s)]
    found = 0
    for start, goal in zip(cells[0:20:2], cells[1:20:2]):
        robots = [cells[-1], cells[-2], (start[0] + 0.5, start[1])] if with_robots else None
        expected, ref = plan(WindowGrid(grid), ext, plan_res, start, goal, robots)
        path, problem = plan(grid, ext, plan_res, start, goal, robots, compiled)
        assert path == expected
        if not compiled:  # the compiled loop keeps its search state in arrays
            assert problem.cost_to_arrive == ref.cost_to_arrive and problem.came_from == ref.came_from
        found += expected is not None
    assert found >= 4 and grid._footprint_free is not None


@pytest.mark.skipif(not lattice_astar.AVAILABLE, reason="numba not installed")
def test_compiled_hypot_is_math_hypot():
    import math
    rng = np.random.default_rng(6)
    a = np.concatenate([rng.normal(0, 1, 20000), rng.uniform(-0.2, 0.2, 20000), [0.0, 0.05, -0.05, 1e-300]])
    b = np.concatenate([rng.normal(0, 1, 20000), rng.uniform(-0.2, 0.2, 20000), [0.0, 0.05, 0.05, 1e300]])
    assert all(lattice_astar.py_hypot(x, y) == math.hypot(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not lattice_astar.AVAILABLE, reason="numba not installed")
def test_compiled_falls_back_when_the_lattice_leaves_the_grid():
    g = make_grid(np.zeros((40, 40), dtype=np.int16))
    problem = AStar((-1.0, -1.0), (3.0, 3.0), (0.5, 0.5), (1.5, 1.5), g, 0.05)  # bounds beyond the 2 x 2 m grid
    assert problem._solve_compiled(g.footprint_lookup()) is None
    assert problem.solve() and problem.path[0] == (0.5, 0.5) and problem.path[-1] == (1.5, 1.5)
