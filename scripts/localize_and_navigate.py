#!/usr/bin/env python3

import time
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String, Int32, Float64, Bool
# from asl_turtlebot.msg import DetectedObject
import tf
import numpy as np
from numpy import linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from enum import Enum
import requests
import os

from dynamic_reconfigure.server import Server
# from asl_turtlebot.cfg import NavigatorConfig

V_PREV_THRES = 0.0001

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3

def wrapToPi(a):
    if isinstance(a, list):
        return [(x+np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    assert(path and k > 2 and k < len(path))
    ########## Code starts here ##########
    t = np.zeros(len(path))
    for ii in range(len(path) - 1):
        dist = np.linalg.norm(np.array(path[ii + 1]) - np.array(path[ii]))  # distance between consecutive points
        t[ii + 1] = dist / V_des + t[ii]  # time at next point assuming constant velocity V_des

    tck_x = scipy.interpolate.splrep(t, np.array(path)[:, 0], k=k, s=alpha)
    tck_y = scipy.interpolate.splrep(t, np.array(path)[:, 1], k=k, s=alpha)

    t_smoothed = np.arange(0, t[-1], dt)
    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed

# A 2D state space grid with a set of rectangular obstacles. The grid is fully deterministic
class DetOccupancyGrid2D(object):
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

class StochOccupancyGrid2D(object):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                window_size, probs, thresh=0.5, robot_d=0.6):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.window_size = 10 # window_size
        # print(window_size)
        self.thresh = thresh
        self.robot_d=robot_d

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def is_free(self, state):
        # combine the probabilities of each cell by assuming independence
        # of each estimation
        x, y = self.snap_to_grid(state)
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        
        # Check if in configuration space
        half_config_size = int(round(self.robot_d/2/self.resolution))
        #print(half_config_size)
        config_x_lower = max(0, grid_x - half_config_size)
        config_y_lower = max(0, grid_y - half_config_size)
        config_x_upper = min(self.width, grid_x + half_config_size)
        config_y_upper = min(self.height, grid_y + half_config_size)
        if np.sum(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]>85):
            #values, counts = np.unique(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper], return_counts=True)
            #print(values)
            #print(counts)
            #print(self.resolution)
            return False  # Not free according to configuration space

        # Now check probabilities
        half_size = int(round((self.window_size-1)/2))
        grid_x_lower = max(0, grid_x - half_size)
        grid_y_lower = max(0, grid_y - half_size)
        grid_x_upper = min(self.width, grid_x + half_size + 1)
        grid_y_upper = min(self.height, grid_y + half_size + 1)        
        
        prob_window = self.probs[grid_y_lower:grid_y_upper, grid_x_lower:grid_x_upper]
        p_total = np.prod(1. - np.maximum(prob_window / 100., 0.))

        return (1. - p_total) < self.thresh         
   
class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, robots_x=None, robots_y=None, robots_d=0.5):
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

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

        self.path = None  # the final path as a list of states

        # Location of other robots in the map
        self.robots_x = robots_x
        self.robots_y = robots_y
        self.robots_d = robots_d  # Diameter of the robot

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
        if self.occupancy.is_free(x) and self.statespace_lo[0] <= x[0] < self.statespace_hi[1] and self.statespace_lo[1] <= x[1] < self.statespace_hi[1]:
            if self.robots_x is not None:
                for ii in range(len(self.robots_x)):
                    if np.linalg.norm(np.array((self.robots_x[ii], self.robots_y[ii])) - np.array(x)) < self.robots_d:
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

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution * round(x[0] / self.resolution), self.resolution * round(x[1] / self.resolution))

    def get_neighbors(self, x):
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
                    x0 += ii * self.resolution  # /(np.linalg.norm(np.array((ii, jj))))
                    x1 += jj * self.resolution  # /(np.linalg.norm(np.array((ii, jj))))
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

    def solve(self):
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
        time_limit = 2
        start = time.time()
        
        while len(self.open_set) > 0:
            # if time.time() - start > time_limit:
                # return False
        
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                
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

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """

    def __init__(self, kpx, kpy, kdx, kdy,
                 V_max=0.5, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8)  # Polynomial coefficients for x(t) and y(t) as
        # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t, self.traj_times, self.traj[:, 0])
        y_d = np.interp(t, self.traj_times, self.traj[:, 1])
        xd_d = np.interp(t, self.traj_times, self.traj[:, 3])
        yd_d = np.interp(t, self.traj_times, self.traj[:, 4])
        xdd_d = np.interp(t, self.traj_times, self.traj[:, 5])
        ydd_d = np.interp(t, self.traj_times, self.traj[:, 6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        if self.V_prev < V_PREV_THRES:
            self.V_prev = np.sqrt(xd_d ** 2 + yd_d ** 2)

        x_dot = self.V_prev * np.cos(th)
        y_dot = self.V_prev * np.sin(th)

        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

        a = u1 * np.cos(th) + u2 * np.sin(th)
        om = -u1 * np.sin(th) / self.V_prev + u2 * np.cos(th) / self.V_prev

        V = self.V_prev + a * dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3,
                 V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

        # rospy.init_node("controller_outputs", anonymous=True)

        # self.pub_alpha = rospy.Publisher('/controller/alpha', Float64, queue_size=10)
        # self.pub_delta = rospy.Publisher('/controller/delta', Float64, queue_size=10)
        # self.pub_rho = rospy.Publisher('/controller/rho', Float64, queue_size=10)
        

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs:
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        rho = np.sqrt((x - self.x_g) ** 2 + (y - self.y_g) ** 2)
        alpha = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - th)
        delta = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - self.th_g)

        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * np.sinc(alpha / np.pi) * np.cos(alpha) * (alpha + self.k3 * delta)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om

class HeadingController:
    """
    pose stabilization controller
    """
    def __init__(self, kp, om_max=1):
        self.kp = kp
        self.om_max = om_max

    def load_goal(self, th_g):
        """
        loads in a new goal position
        """
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        err = wrapToPi(self.th_g - th)
        om = self.kp*err

        # apply control limits
        V = 0
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self, server_url='http://192.168.50.2:8000/graphql'):
        rospy.init_node("mattbot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        self.is_localized = False

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        self.server_url = server_url
        self.stopped_robot_location_query = """
                                            {
                                                stoppedRobotPositions {
                                                    id
                                                    x
                                                    y
                                                    theta 
                                                }
                                            }
                                            """
                                            
        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 50

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.4  # maximum velocity
        self.om_max = 2.5  # maximum angular velocity
        self.om_heading = 1.3  # angular velocity for heading controller

        self.v_des = 0.2  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.01
        self.at_thresh = 0.01
        self.at_thresh_theta = 0.05
        self.theta_goal_thresh = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.spline_deg = 3  # cubic spline
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 1
        self.kpy = 1
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)
        self.heading_controller2 = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel_mux/input/nav_vel", Twist, queue_size=10)

        #Publishes current state of robot (IDLE, ALIGN, etc)
        self.state_pub = rospy.Publisher("/robot_mode", Int32, queue_size=10)

        self.trans_listener = tf.TransformListener()

        # self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber("/external_goal", Pose2D, self.external_goal_callback)
        self.localized_sub = rospy.Subscriber("/localized", Bool, self.localized_callback)

        self.has_stopped = False

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            # rospy.logout(f"New command nav received:\n{data}")
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                pass
                # rospy.loginfo("replanning because of new map")
                # self.replan()  # new map, need to replan

    def rviz_goal_callback(self, msg):
        print("RViz goal received")
        origin_frame = "map"
        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            x_g_proposed = nav_pose_origin.pose.position.x
            y_g_proposed = nav_pose_origin.pose.position.y

            if not self.occupancy.is_free((x_g_proposed, y_g_proposed)):
                rospy.loginfo("Not a valid goal")
                return
            
            self.x_g = x_g_proposed
            self.y_g = y_g_proposed

            quaternion = (nav_pose_origin.pose.orientation.x, nav_pose_origin.pose.orientation.y, nav_pose_origin.pose.orientation.z, nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
            print(self.x_g)
            print(self.y_g)
            print(self.theta_g)
            self.replan()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("RVIZ Goal exception:")
            print(e)
    
    def external_goal_callback(self, msg):
        x_g_proposed = msg.x
        y_g_proposed = msg.y

        if not self.occupancy.is_free((x_g_proposed, y_g_proposed)):
            rospy.loginfo("Not a valid goal")
            return
        
        self.x_g = x_g_proposed
        self.y_g = y_g_proposed
        self.theta_g = msg.theta
        self.replan()

    def localized_callback(self, msg):
        self.is_localized = msg.data

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )
    
    def aligned_goal(self):
        """
        returns whether robot is aligned with goal direction
        """
        return (
            abs(wrapToPi(self.theta - self.theta_g)) < self.theta_goal_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode
        self.state_pub.publish(self.mode.value)

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher, times=None):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        current_time = rospy.get_rostime()
        path_msg.header.stamp = current_time
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            if times is not None:
                pose_st.header.stamp = rospy.Duration(times[i]) + current_time
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            # V, om = self.pose_controller.compute_control(
            #     self.x, self.y, self.theta, t
            # )
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):

        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def get_stopped_robot_locations(self):
        # Query using graphql
        response = requests.post(self.server_url, json={'query': self.stopped_robot_location_query})
        data = response.json()

        # Extract the data
        position_data = data.get('data', {}).get('stoppedRobotPositions', {})
        x = []
        y = []
        theta = []

        my_id = int(os.environ.get('ROBOT_ID'))
        for robot in position_data:
            if robot.get('id') != my_id:
                x.append(robot.get('x'))
                y.append(robot.get('y'))
                theta.append(robot.get('theta'))
        return x, y, theta

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        robots_x, robots_y, robots_theta = self.get_stopped_robot_locations()
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
            robots_x=robots_x,
            robots_y=robots_y
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success and self.mode == Mode.IDLE:
            rospy.loginfo("Planning failed")
            self.x_g += np.random.normal(0,0.05)
            self.y_g += np.random.normal(0,0.05)
            self.replan()
            return
        else:
            rospy.loginfo("Planning Succeeded")
            planned_path = problem.path

        # Check whether path is too short
        if planned_path == None:
            return
        elif len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.heading_controller.load_goal(self.theta_g)
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        t_new, traj_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_deg, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                # self.publish_smoothed_path(
                #     traj_new, self.nav_smoothed_path_rej_pub
                # )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub, times=t_new)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def localize(self):
        """
        Rotates slowly to localize the robot
        """
        rate = rospy.Rate(10)  # 10 Hz
        while not self.is_localized:
            # rotate until we get a valid position
            cmd_vel = Twist()
            cmd_vel.angular.z = 1
            self.nav_vel_pub.publish(cmd_vel)

            rate.sleep()

        # Display message that we have localized
        rospy.loginfo("Navigator: localized")

        # Unsubscribe from the localized topic
        self.is_localized = True
        self.localized_sub.unregister()
        self.localized_sub = None

        # Now that we are localized, we can stop the robot
        cmd_vel = Twist()
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
        
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.heading_controller.load_goal(self.theta_g)
                    print("Setting theta goal to", self.theta_g)
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                # Reached goal: forget goal coordinates and stop
                if self.aligned_goal():
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)

            self.publish_control()
            #rate.sleep()
            time.sleep(0.01)


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(5)  # Give time for everything to set up
    nav.localize()  # rotate to localize
    nav.run()  # run the main loop
