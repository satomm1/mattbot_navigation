import math
import numpy as np
import time
from queue import PriorityQueue

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, robots_x=None, robots_y=None, obj_x=None, obj_y=None, obj_d=None, robots_d=0.4, max_plan_time_sec=15.0):
        self.statespace_lo = np.array(statespace_lo)  # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)  # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy  # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution  # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)  # initial state
        self.x_goal = self.snap_to_grid(x_goal)  # goal state

        self.closed_set = set()  # the set containing the states that have been visited
        self.open_set = set()  # the set containing the states that are condidate for future expension
        self.came_from = {}  # dictionary keeping track of each state's parent to reconstruct the path
        self.est_cost_through = {}
        self.cost_to_arrive = {}

        self.priority_queue = PriorityQueue()
        self.priority_queue.put((self.manhattan_distance(self.x_init, self.x_goal), self.x_init))

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

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
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if self.occupancy.is_free(x) and self.statespace_lo[0] <= x[0] < self.statespace_hi[0] and self.statespace_lo[1] <= x[1] < self.statespace_hi[1]:
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
        else:
            return False
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        ########## Code starts here ##########
        return math.hypot(x1[0] - x2[0], x1[1] - x2[1])
        ########## Code ends here ##########

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
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for ii in [1, 0, -1]:
            for jj in [1, 0, -1]:
                if ii != 0 or jj != 0:
                    x0 = x[0]
                    x1 = x[1]
                    x0 += ii * self.resolution * step_resolution  # /(np.linalg.norm(np.array((ii, jj))))
                    x1 += jj * self.resolution * step_resolution  # /(np.linalg.norm(np.array((ii, jj))))
                    state = self.snap_to_grid((x0, x1))
                    if self.is_free(state):
                        neighbors.append(state)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def interpolate_path_linear(self, path, step_resolution=2):
        """Linearly interpolate between path points"""
        path_array = np.array(path)
        new_path = []
        for i in range(len(path_array) - 1):
            start = path_array[i]
            end = path_array[i + 1]
            num_to_add = step_resolution - 1
            
            new_path.append(self.snap_to_grid(start))
            for j in range(num_to_add):
                t = (j + 1) / (num_to_add + 1)
                interpolated_point = start + t * (end - start)
                new_path.append(self.snap_to_grid(interpolated_point))
        new_path.append(self.snap_to_grid(path_array[-1]))
        return new_path

    def reconstruct_path(self, step_resolution=1):
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

        # if step_resolution > 1:
        #     return self.interpolate_path_linear(list(reversed(path)), step_resolution=step_resolution)

        return list(reversed(path))

    def solve_old(self, step_resolution=1):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        time_limit = self.max_plan_time_sec

        print("step resolution: ", step_resolution)

        start = time.time()        
        print(self.x_init)
        while len(self.open_set) > 0:
            if time.time() - start > time_limit:
                print("A* took too long")
                return False
        
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path(step_resolution=step_resolution)
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current, step_resolution=step_resolution):

                # if x_neigh not in self.closed_set:
                #     continue
                # tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                # if x_neigh not in self.open_set:
                #     self.open_set.add(x_neigh)
                # elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                #     continue
                # self.came_from[x_neigh] = x_current
                # self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                # self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.cost_to_arrive or tentative_cost_to_arrive < self.cost_to_arrive[x_neigh]:
                    self.open_set.add(x_neigh)
                    self.came_from[x_neigh] = x_current
                    self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                    self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False
        ########## Code ends here ##########

    def solve(self, step_resolution=1):
        time_limit = self.max_plan_time_sec

        t_start = time.time()
        while self.priority_queue.qsize() > 0:
            current_cost, x_current = self.priority_queue.get()

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
                    self.priority_queue.put(
                        (current_cost + edge + self.h(x_neigh) - h_current, x_neigh)
                    )
        return False
