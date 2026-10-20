import heapq
import math
import numpy as np
import time

from navigation_utils import lattice_astar

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, robots_x=None, robots_y=None, obj_x=None, obj_y=None, obj_d=None, robots_d=0.4, max_plan_time_sec=15.0):
        self.statespace_lo = np.array(statespace_lo)  # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)  # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy  # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution  # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)  # initial state
        self.x_goal = self.snap_to_grid(x_goal)  # goal state

        # Cached planner bounds + 8-connected offsets (world meters).
        self._x_min = float(self.statespace_lo[0])
        self._x_max = float(self.statespace_hi[0])
        self._y_min = float(self.statespace_lo[1])
        self._y_max = float(self.statespace_hi[1])
        r = float(self.resolution)
        self._neighbor_offsets = tuple(
            (i * r, j * r)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if i or j
        )

        self.closed_set = set()  # the set containing the states that have been visited
        self.came_from = {}  # dictionary keeping track of each state's parent to reconstruct the path
        self.cost_to_arrive = {}

        self.priority_queue = []
        heapq.heappush(self.priority_queue, (self.h(self.x_init), self.x_init))

        self.cost_to_arrive[self.x_init] = 0

        self.path = None  # the final path as a list of states

        # Location of other robots in the map
        self.robots_x = robots_x
        self.robots_y = robots_y
        self.robots_d = robots_d  # Diameter of the robot

        self.obj_x = obj_x
        self.obj_y = obj_y
        self.obj_d = obj_d
        self.max_plan_time_sec = float(max_plan_time_sec)

    def is_free(self, x):
        """
        Checks if a given state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        """
        # Bounds first (cheap) before occupancy / agent checks.
        if not (self._x_min <= x[0] < self._x_max and self._y_min <= x[1] < self._y_max):
            return False
        if not self.occupancy.is_free(x):
            return False
        if self.robots_x is not None:
            robots_d2 = self.robots_d * self.robots_d
            for ii in range(len(self.robots_x)):
                dx = self.robots_x[ii] - x[0]
                dy = self.robots_y[ii] - x[1]
                if dx * dx + dy * dy < robots_d2:
                    return False
            for ii in range(len(self.obj_x)):
                r = self.obj_d[ii] / 2 + self.robots_d / 2
                dx = self.obj_x[ii] - x[0]
                dy = self.obj_y[ii] - x[1]
                if dx * dx + dy * dy < r * r:
                    return False
        return True

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        return math.hypot(x1[0] - x2[0], x1[1] - x2[1])

    def manhattan_distance(self, x1, x2):
        """
        Computes the Manhattan distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Manhattan distance
        """
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    def h(self, x):
        return self.manhattan_distance(x, self.x_goal)

    def cost(self, x1, x2):
        return self.distance(x1, x2)

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution * round(x[0] / self.resolution), self.resolution * round(x[1] / self.resolution))

    def get_neighbors(self, x, step_resolution=1):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Output:
            List of neighbors that are free, as a list of TUPLES
        """
        neighbors = []
        x0, y0 = x[0], x[1]
        if step_resolution == 1:
            offsets = self._neighbor_offsets
        else:
            s = self.resolution * step_resolution
            offsets = (
                (i * s, j * s)
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if i or j
            )

        xmin, xmax = self._x_min, self._x_max
        ymin, ymax = self._y_min, self._y_max
        res = self.resolution
        for dx, dy in offsets:
            nx = x0 + dx
            ny = y0 + dy
            if nx < xmin or nx >= xmax or ny < ymin or ny >= ymax:
                continue
            # Re-snap to keep a stable float lattice (0.05 etc. are not binary-exact).
            state = (res * round(nx / res), res * round(ny / res))
            if self.is_free(state):
                neighbors.append(state)
        return neighbors

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self, step_resolution=1):
        footprint_lookup = getattr(self.occupancy, "footprint_lookup", None)
        if step_resolution == 1 and footprint_lookup is not None:
            free = footprint_lookup()
            solved = self._solve_compiled(free) if self.use_compiled else None
            if solved is not None:
                return solved
            return self._solve_on_lookup(free)
        time_limit = self.max_plan_time_sec

        t_start = time.time()  # wall clock: time_limit is a compute budget
        while self.priority_queue:
            current_cost, x_current = heapq.heappop(self.priority_queue)

            # Lazy PQ: skip stale entries for nodes already expanded.
            if x_current in self.closed_set:
                continue

            if x_current == self.x_goal:
                t_end = time.time()  # wall clock: compute time
                self.path = self.reconstruct_path()
                print(f"A* found a path in {t_end - t_start:.2f} seconds.")
                return True

            if time.time() - t_start > time_limit:  # wall clock: compute budget
                print("A* took too long.")
                return False

            self.closed_set.add(x_current)
            h_current = self.h(x_current)

            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue

                edge = self.distance(x_current, x_neigh)
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + edge

                if x_neigh not in self.cost_to_arrive or tentative_cost_to_arrive < self.cost_to_arrive[x_neigh]:
                    self.came_from[x_neigh] = x_current
                    self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                    heapq.heappush(
                        self.priority_queue,
                        (current_cost + edge + self.h(x_neigh) - h_current, x_neigh),
                    )
        return False

    use_compiled = True  # the numba loop when numba is available (lattice_astar.AVAILABLE)

    def _solve_compiled(self, free):
        """_solve_on_lookup in compiled code (navigation_utils.lattice_astar): the same path. Sets only self.path
        (not closed_set / came_from / cost_to_arrive). None when it cannot be used."""
        if self.robots_x is not None:
            robots = list(zip(self.robots_x, self.robots_y))
            objects = [(self.obj_x[ii], self.obj_y[ii], self.obj_d[ii] / 2 + self.robots_d / 2)
                       for ii in range(len(self.obj_x))]
            robots_d2 = self.robots_d * self.robots_d
        else:
            robots, objects, robots_d2 = [], [], 0.0
        t_start = time.time()  # wall clock: compute time
        result = lattice_astar.solve(free, self.occupancy, self.resolution, self._neighbor_offsets,
                                     (self._x_min, self._x_max, self._y_min, self._y_max), self.x_init, self.x_goal,
                                     robots, robots_d2, objects, self.max_plan_time_sec)
        if result is None:
            return None
        status, path = result
        if status == lattice_astar.FOUND:
            self.path = path
            t_end = time.time()  # wall clock: compute time
            print(f"A* found a path in {t_end - t_start:.2f} seconds.")
            return True
        if status == lattice_astar.TIMED_OUT:
            print("A* took too long.")
        return False

    def _solve_on_lookup(self, free):
        """solve() with get_neighbors / is_free / h / distance inlined and the occupancy's precomputed footprint
        lookup (StochOccupancyGrid2D.footprint_lookup) in place of occupancy.is_free. The same float states,
        expansion order, costs and tie-breaks as solve(), so the same path; only faster (A* dominates planning).
        """
        occ = self.occupancy
        width, height = occ.width, occ.height
        free_cells = np.ascontiguousarray(free, dtype=bool).tobytes()  # row-major, 1 = free
        ox, oy, gres = occ.origin_x, occ.origin_y, occ.resolution
        # occupancy.is_free re-snaps to its own resolution; a no-op for states already on the same lattice.
        resnap = gres != self.resolution
        occupancy_is_free = occ.is_free
        res = self.resolution
        offsets = self._neighbor_offsets
        xmin, xmax, ymin, ymax = self._x_min, self._x_max, self._y_min, self._y_max
        gx_goal, gy_goal = self.x_goal
        if self.robots_x is not None:
            robots_d2 = self.robots_d * self.robots_d
            robots = list(zip(self.robots_x, self.robots_y))
            objects = [(self.obj_x[ii], self.obj_y[ii], self.obj_d[ii] / 2 + self.robots_d / 2)
                       for ii in range(len(self.obj_x))]
        else:
            robots = objects = None

        closed_set = self.closed_set
        came_from = self.came_from
        cost_to_arrive = self.cost_to_arrive
        queue = self.priority_queue
        heappop, heappush, hypot = heapq.heappop, heapq.heappush, math.hypot
        time_limit = self.max_plan_time_sec

        t_start = time.time()  # wall clock: time_limit is a compute budget
        while queue:
            current_cost, x_current = heappop(queue)
            if x_current in closed_set:
                continue

            if x_current == self.x_goal:
                t_end = time.time()  # wall clock: compute time
                self.path = self.reconstruct_path()
                print(f"A* found a path in {t_end - t_start:.2f} seconds.")
                return True

            if time.time() - t_start > time_limit:  # wall clock: compute budget
                print("A* took too long.")
                return False

            closed_set.add(x_current)
            x0, y0 = x_current
            h_current = abs(x0 - gx_goal) + abs(y0 - gy_goal)
            cost_current = cost_to_arrive[x_current]

            for dx, dy in offsets:
                # get_neighbors
                nx = x0 + dx
                ny = y0 + dy
                if nx < xmin or nx >= xmax or ny < ymin or ny >= ymax:
                    continue
                sx = res * round(nx / res)
                sy = res * round(ny / res)
                # is_free
                if not (xmin <= sx < xmax and ymin <= sy < ymax):
                    continue
                if resnap:
                    cx, cy = gres * round(sx / gres), gres * round(sy / gres)
                else:
                    cx, cy = sx, sy
                grid_x = int((cx - ox) / gres)
                grid_y = int((cy - oy) / gres)
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    if not free_cells[grid_y * width + grid_x]:
                        continue
                elif not occupancy_is_free((sx, sy)):
                    continue
                if robots is not None:
                    blocked = False
                    for rx, ry in robots:
                        ddx = rx - sx
                        ddy = ry - sy
                        if ddx * ddx + ddy * ddy < robots_d2:
                            blocked = True
                            break
                    if not blocked:
                        for qx, qy, r in objects:
                            ddx = qx - sx
                            ddy = qy - sy
                            if ddx * ddx + ddy * ddy < r * r:
                                blocked = True
                                break
                    if blocked:
                        continue
                x_neigh = (sx, sy)
                # solve
                if x_neigh in closed_set:
                    continue
                edge = hypot(x0 - sx, y0 - sy)
                tentative_cost_to_arrive = cost_current + edge
                if x_neigh not in cost_to_arrive or tentative_cost_to_arrive < cost_to_arrive[x_neigh]:
                    came_from[x_neigh] = x_current
                    cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                    heappush(queue, (current_cost + edge + (abs(sx - gx_goal) + abs(sy - gy_goal)) - h_current,
                                     x_neigh))
        return False
