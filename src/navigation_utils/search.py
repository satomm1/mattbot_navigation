import heapq
import math
import numpy as np
import time

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
        time_limit = self.max_plan_time_sec

        t_start = time.time()
        while self.priority_queue:
            current_cost, x_current = heapq.heappop(self.priority_queue)

            # Lazy PQ: skip stale entries for nodes already expanded.
            if x_current in self.closed_set:
                continue

            if x_current == self.x_goal:
                t_end = time.time()
                self.path = self.reconstruct_path()
                print(f"A* found a path in {t_end - t_start:.2f} seconds.")
                return True

            if time.time() - t_start > time_limit:
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
