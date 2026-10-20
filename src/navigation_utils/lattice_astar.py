"""Compiled (numba) A* inner loop for search.AStar on a StochOccupancyGrid2D footprint lookup.

It repeats AStar._solve_on_lookup's float arithmetic step for step (the same neighbour order, snapped states,
costs, priorities and (priority, x, y) heap tie-breaks), so it returns the same path; only the dicts / sets
become arrays over the planner's lattice. numba is optional: without it (or with MATTBOT_ASTAR_NUMBA=0)
AVAILABLE is False and AStar uses the Python loop.
"""

import os
import time

import numpy as np

try:
    if os.environ.get("MATTBOT_ASTAR_NUMBA", "1") == "0":
        raise ImportError("disabled by MATTBOT_ASTAR_NUMBA=0")
    import numba
    from numba import njit
except ImportError:  # pragma: no cover - depends on the container
    numba = None
    AVAILABLE = False
else:
    AVAILABLE = True

FOUND, EXHAUSTED, TIMED_OUT = 1, 0, -1
TIME_CHECK_EVERY = 1024  # pops between wall-clock checks (time.time() from compiled code is not free)


def lattice_axis(res, lo, hi, cell_res, cell_origin, resnap, n_cells):
    """Planner lattice along one axis: index i is the state res * (i + m0).

    Returns (m0, values, cells, ok): values[i] is that state's float (res * m, as res * round(v / res) gives
    it), cells[i] its occupancy-grid index (StochOccupancyGrid2D.is_free's int((v - origin) / resolution)
    after its own snap), and ok whether every state inside [lo, hi) maps into the grid (else the compiled
    loop cannot be used: is_free's off-grid window check is Python).
    """
    m0 = int(np.floor(lo / res)) - 1
    m1 = int(np.ceil(hi / res)) + 1
    m = np.arange(m0, m1 + 1, dtype=np.int64)
    values = res * m.astype(np.float64)
    snapped = cell_res * np.rint(values / cell_res) if resnap else values
    cells = ((snapped - cell_origin) / cell_res).astype(np.int64)  # int() truncates toward zero, as astype does
    inside = (values >= lo) & (values < hi)
    ok = bool(np.all((cells[inside] >= 0) & (cells[inside] < n_cells)))
    return m0, values, cells, ok


def _now():
    return time.time()  # wall clock: time_limit is a compute budget


if AVAILABLE:
    import heapq

    @njit(cache=True)
    def _wall_time():
        with numba.objmode(t="float64"):
            t = _now()
        return t

    @njit(cache=True)
    def py_hypot(x, y):
        """math.hypot(x, y) as CPython 3.8 computes it (Modules/mathmodule.c vector_norm), bit for bit; libm's
        hypot can differ in the last bit, which would change tie-breaks."""
        x = abs(x)
        y = abs(y)
        m = max(x, y)
        if np.isinf(m):
            return m
        if np.isnan(x) or np.isnan(y):
            return np.nan
        if m == 0.0:
            return m
        csum = 1.0
        frac = 0.0
        for v in (x, y):
            v = v / m
            v = v * v
            oldcsum = csum
            csum += v
            frac += (oldcsum - csum) + v
        return m * np.sqrt(csum - 1.0 + frac)

    @njit(cache=True)
    def _solve(free, xs, ys, cols, rows, mx0, my0, res, offsets, xmin, xmax, ymin, ymax,
               ix_init, iy_init, x_goal, y_goal, robots, robots_d2, objects, time_limit):
        nx_cells = xs.shape[0]
        ny_cells = ys.shape[0]
        closed = np.zeros((nx_cells, ny_cells), dtype=np.bool_)
        cost = np.full((nx_cells, ny_cells), np.inf)
        parent = np.full((nx_cells, ny_cells, 2), -1, dtype=np.int32)

        x_init = xs[ix_init]
        y_init = ys[iy_init]
        cost[ix_init, iy_init] = 0.0
        h_init = abs(x_init - x_goal) + abs(y_init - y_goal)
        queue = [(h_init, x_init, y_init, ix_init, iy_init)]
        n_robots = robots.shape[0]
        n_objects = objects.shape[0]
        n_offsets = offsets.shape[0]

        t_start = _wall_time()
        pops = 0
        while len(queue) > 0:
            current_cost, x0, y0, ix, iy = heapq.heappop(queue)
            if closed[ix, iy]:
                continue
            if x0 == x_goal and y0 == y_goal:
                n = 1
                cx, cy = ix, iy
                while cx != ix_init or cy != iy_init:
                    px = parent[cx, cy, 0]
                    py = parent[cx, cy, 1]
                    cx, cy = px, py
                    n += 1
                path = np.empty((n, 2))
                cx, cy = ix, iy
                for k in range(n - 1, -1, -1):
                    path[k, 0] = xs[cx]
                    path[k, 1] = ys[cy]
                    if k > 0:
                        px = parent[cx, cy, 0]
                        py = parent[cx, cy, 1]
                        cx, cy = px, py
                return FOUND, path
            pops += 1
            if pops % TIME_CHECK_EVERY == 0 and _wall_time() - t_start > time_limit:
                return TIMED_OUT, np.empty((0, 2))

            closed[ix, iy] = True
            h_current = abs(x0 - x_goal) + abs(y0 - y_goal)
            cost_current = cost[ix, iy]
            for k in range(n_offsets):
                nx = x0 + offsets[k, 0]
                ny = y0 + offsets[k, 1]
                if nx < xmin or nx >= xmax or ny < ymin or ny >= ymax:
                    continue
                mx = np.rint(nx / res)
                my = np.rint(ny / res)
                sx = res * mx
                sy = res * my
                if not (xmin <= sx and sx < xmax and ymin <= sy and sy < ymax):
                    continue
                jx = int(mx) - mx0
                jy = int(my) - my0
                if not free[rows[jy], cols[jx]]:
                    continue
                blocked = False
                for q in range(n_robots):
                    ddx = robots[q, 0] - sx
                    ddy = robots[q, 1] - sy
                    if ddx * ddx + ddy * ddy < robots_d2:
                        blocked = True
                        break
                if not blocked:
                    for q in range(n_objects):
                        ddx = objects[q, 0] - sx
                        ddy = objects[q, 1] - sy
                        r = objects[q, 2]
                        if ddx * ddx + ddy * ddy < r * r:
                            blocked = True
                            break
                if blocked:
                    continue
                if closed[jx, jy]:
                    continue
                edge = py_hypot(x0 - sx, y0 - sy)
                tentative = cost_current + edge
                if tentative < cost[jx, jy]:  # inf when not yet reached, as "not in cost_to_arrive"
                    parent[jx, jy, 0] = ix
                    parent[jx, jy, 1] = iy
                    cost[jx, jy] = tentative
                    heapq.heappush(queue, (current_cost + edge + (abs(sx - x_goal) + abs(sy - y_goal)) - h_current,
                                           sx, sy, jx, jy))
        return EXHAUSTED, np.empty((0, 2))


def solve(free, occupancy, resolution, offsets, bounds, x_init, x_goal, robots, robots_d2, objects, time_limit):
    """Run the compiled loop; returns (status, path as a list of (x, y)) or None when it cannot be used."""
    if not AVAILABLE:
        return None
    xmin, xmax, ymin, ymax = bounds
    res = float(resolution)
    gres = float(occupancy.resolution)
    resnap = gres != res
    mx0, xs, cols, ok_x = lattice_axis(res, xmin, xmax, gres, occupancy.origin_x, resnap, occupancy.width)
    my0, ys, rows, ok_y = lattice_axis(res, ymin, ymax, gres, occupancy.origin_y, resnap, occupancy.height)
    if not (ok_x and ok_y):
        return None
    ix_init = int(round(x_init[0] / res)) - mx0
    iy_init = int(round(x_init[1] / res)) - my0
    if not (0 <= ix_init < xs.shape[0] and 0 <= iy_init < ys.shape[0]):
        return None
    if xs[ix_init] != x_init[0] or ys[iy_init] != x_init[1]:
        return None  # not on the lattice (AStar snaps x_init, so this does not happen)
    status, path = _solve(np.ascontiguousarray(free, dtype=np.bool_), xs, ys, np.clip(cols, 0, occupancy.width - 1),
                          np.clip(rows, 0, occupancy.height - 1), mx0, my0, res,
                          np.asarray(offsets, dtype=np.float64).reshape(-1, 2),
                          float(xmin), float(xmax), float(ymin), float(ymax), ix_init, iy_init,
                          float(x_goal[0]), float(x_goal[1]),
                          np.asarray(robots, dtype=np.float64).reshape(-1, 2), float(robots_d2),
                          np.asarray(objects, dtype=np.float64).reshape(-1, 3), float(time_limit))
    return status, [(float(x), float(y)) for x, y in path]
