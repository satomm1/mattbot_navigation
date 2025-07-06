#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String, Int32, Float64, Bool
from visualization_msgs.msg import Marker, MarkerArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
from mattbot_dds.msg import AgentPath, AgentLocation
import tf

import time
import numpy as np
from numpy import linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from enum import Enum
import requests
import os

from navigation_utils import TrajectoryTracker, PoseController, HeadingController, wrapToPi, StochOccupancyGrid2D, AStar, compute_smoothed_traj

V_PREV_THRES = 0.0001

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

PERSON_STOP_DISTANCE = 1.6  # distance to closest person at which we stop the robot
PERSON_SLOW_DISTANCE = 2.5  # distance to closest person at which we slow down the robot

class Mode(Enum):
    IDLE = 0        # not moving, waiting for a goal
    LOCALIZING = 1  # localizing the robot
    ALIGN = 2       # aligning to start heading of the path
    TRACK = 3       # tracking the path (following the trajectory)
    PARK = 4        # parking the robot (moving to a specific pose)
    BACKING = 5     # backing up when stuck
    WAITING_FOR_INIT = 6

class Navigator:
    """
    This node handles point to point mattbot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("mattbot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        self.is_localized = False
        self.received_initial_pose = False
        self.localize_begin_time = None

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

        self.person_occupancy = None
        self.robot_stopped_by_person = False
        self.person_in_path = False
        self.person_list1 = []
        self.person_list2 = []
        self.person_list3 = []
        self.stopped_for_person_time = rospy.get_rostime().to_sec()

        self.other_agents_goals = dict()
        self.other_agents_at_goal = []
        self.other_agents_static = []  # list of agent ID's that are static (not moving)
        self.other_agents_locations = dict()  # agent ID -> [x, y, theta, time]
                                            
        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 500

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        self.times_planned_failed = 0

        # Robot limits
        self.v_max = 0.7  # maximum velocity
        self.om_max = 3  # maximum angular velocity
        self.om_heading = 1.3  # angular velocity for heading controller

        self.v_des = 0.4  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.1
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
        self.kp_th = 1.5
        self.om_prev = 0.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)
        self.heading_controller2 = HeadingController(self.kp_th, self.om_max)

        # Data structures to hold the detected objects
        self.detected_objects = []
        self.current_plan = []
        self.unsmoothed_plan = []

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel_mux/input/nav_vel", Twist, queue_size=10)

        #Publishes current state of robot (IDLE, ALIGN, etc)
        self.state_pub = rospy.Publisher("/robot_mode", Int32, queue_size=10)

        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)

        self.trans_listener = tf.TransformListener()

        # Get map parameter to determine what map to use
        map_name = rospy.get_param('map_name', '/map')

        # Distance to closest person --- determines if we should slow down or stop
        self.distance_to_person = np.inf

        rospy.Subscriber(map_name, OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber("/external_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/voice_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/agent_location", AgentLocation, self.agent_location_callback)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_callback)
        self.localized_pub = rospy.Publisher("/localized", Bool, queue_size=10)

        self.has_stopped = False

        self.waypoints = []
        self.backing_start_time = 0

        self.switch_mode(Mode.WAITING_FOR_INIT)

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

    def external_goal_callback(self, msg):
        """
        Callback for external goals, such as from voice commands or from DDS
        """
        # Make sure the goal is different from the current goal
        if (
            self.x_g is not None
            and self.y_g is not None
            and self.theta_g is not None
            and (msg.x == self.x_g and msg.y == self.y_g and msg.theta == self.theta_g)
        ):
            rospy.loginfo("External goal is the same as current goal, ignoring")
            return

        # Stop the bot
        if self.mode != Mode.IDLE:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)

            self.switch_mode(Mode.IDLE)

        # Make sure we have an occupancy grid and that the goal is valid
        if self.occupancy is not None and not self.occupancy.is_free((msg.x, msg.y)):
            rospy.loginfo("Not a valid goal")
            return
        
        # Update the goal
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.replan()

    def rviz_goal_callback(self, msg):
        """
        Callback for RViz goals, transforms the goal to the map frame and checks if it is valid
        """
        
        print("RViz goal received")
        origin_frame = "map"
        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            x_g_proposed = nav_pose_origin.pose.position.x
            y_g_proposed = nav_pose_origin.pose.position.y            
            quaternion = (nav_pose_origin.pose.orientation.x, nav_pose_origin.pose.orientation.y, nav_pose_origin.pose.orientation.z, nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)

            # Send goal to the external goal callback for processing
            new_msg = Pose2D()
            new_msg.x = x_g_proposed
            new_msg.y = y_g_proposed
            new_msg.theta = euler[2]
            self.external_goal_callback(new_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("RVIZ Goal exception:")
            print(e)

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

            if self.person_occupancy is None:
                self.person_occupancy = StochOccupancyGrid2D(
                    self.map_resolution,
                    self.map_width,
                    self.map_height,
                    self.map_origin[0],
                    self.map_origin[1],
                    5,
                    np.zeros((self.map_width * self.map_height,)),  # initialize with zeros     
                )

    def initial_pose_callback(self, msg):
        """
        Callback for initial pose, sets the robot's position and orientation
        """        
        rospy.loginfo("Navigator: Initial pose received")
        self.received_initial_pose = True

    def agent_location_callback(self, msg):
        """
        Callback for /agent_location topic.
        Updates the location of other agents and checks if they are static.
        """
        current_time = rospy.get_rostime()
        agent_id = msg.agentID.data

        # Keep track of the previous pose of the agent (if it exists)
        prev_pose = None
        if agent_id in self.other_agents_locations:
            prev_pose = self.other_agents_locations[agent_id]

        # Get agent's pose and time
        agent_pose = msg.pose
        x = agent_pose.position.x
        y = agent_pose.position.y
        quaternion = (agent_pose.orientation.x, agent_pose.orientation.y, agent_pose.orientation.z, agent_pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        theta = euler[2]
        self.other_agents_locations[agent_id] = [x, y, theta, current_time]

        # Check if agent is moving
        if prev_pose is not None:
            if np.linalg.norm(np.array([x - prev_pose[0], y - prev_pose[1]]) < 0.1) and np.abs(theta - prev_pose[2]) < 0.1:
                if agent_id not in self.other_agents_static:
                    self.other_agents_static.append(agent_id)
            else: 
                if agent_id in self.other_agents_static:
                    self.other_agents_static.remove(agent_id)

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
    
    def aligned_start(self):
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

    def person_intersect_path(self):
        """
        Use the self.person_occupancy to check if the path intersects with a person
        """
        person_probs = self.person_occupancy.get_probs()
        path = self.current_plan

        for point in path:
            grid_x = int((point[0] - self.map_origin[0]) / self.map_resolution)
            grid_y = int((point[1] - self.map_origin[1]) / self.map_resolution)
            if (0 <= grid_x < self.map_width) and (0 <= grid_y < self.map_height):
                if person_probs[grid_y, grid_x] > 0.5:  # Assuming a threshold of 0.5
                    return True

        return False
    
    def detected_objects_callback(self, msg):
        """
        receives detected objects, only looks at people and stores their location in the 
        self.person_occupancy occupancy grid map
        """

        # Each detection stays valid for 3 frames
        self.person_list3 = self.person_list2  # 2 frames ago
        self.person_list2 = self.person_list1
        self.person_list1 = []  # Most recent --- now

        # Get the distance to the closest person
        closest_person_dist = np.inf
        person_probs = np.zeros((self.map_height, self.map_width))

        for obj in msg.objects:
            if obj.class_name != "person":
                continue

            x = obj.pose.position.x
            y = obj.pose.position.y
            self.person_list1.append((x, y))

        for (x,y) in self.person_list1 + self.person_list2 + self.person_list3: 
            radius = 0.3  # assume person occupies a circle of radius 0.3m

            # Get x,y coordinates in terms of gird coordinates
            grid_x = int((x - self.map_origin[0]) / self.map_resolution)
            grid_y = int((y - self.map_origin[1]) / self.map_resolution)

            # make sure coordinates are within map
            grid_x = np.clip(grid_x, 0, self.map_width - 1)
            grid_y = np.clip(grid_y, 0, self.map_height - 1)

            # Update the person occupancy grid
            # TODO Vectorize this
            for i in range(-self.person_occupancy.window_size//2, self.person_occupancy.window_size//2 + 1):
                for j in range(-self.person_occupancy.window_size//2, self.person_occupancy.window_size//2 + 1):
                    if (0 <= grid_x + i < self.map_width) and (0 <= grid_y + j < self.map_height):
                        dist = np.sqrt(i**2 + j**2) * self.map_resolution
                        if dist <= radius:
                            person_probs[grid_y + j, grid_x + i] = 1.0  # Mark as occupied

            dist_to_person = np.linalg.norm(np.array([x - self.x, y - self.y]))

            if dist_to_person < closest_person_dist:
                closest_person_dist = dist_to_person
        self.distance_to_person = closest_person_dist

        self.person_occupancy.update(person_probs)

        self.person_in_path = self.person_intersect_path()

    def modify_velocity_for_person(self, V, om):
        """
        Modifies the velocity based on the distance to the closest person.
        If the distance is less than a threshold, it slows down or stops the robot.
        """
        if self.distance_to_person < PERSON_STOP_DISTANCE and self.person_in_path:
            # Stop
            V = 0.0
            om = 0.0
            
            # Keep track of how long we've been stopped
            if not self.robot_stopped_by_person:
                self.stopped_for_person_time = rospy.get_rostime().to_sec()
                print("Stopping for person")

            self.robot_stopped_by_person = True
        elif self.distance_to_person < PERSON_SLOW_DISTANCE:
            # Slow down
            V *= 0.5
            om *= 0.5
        elif self.robot_stopped_by_person:
            # If we were stopped by a person, we can start moving again
            self.robot_stopped_by_person = False

            print("Resuming motion after stopping for person")

            self.replan()

        return V, om
    
    def path_still_valid(self, path):
        for point in path:
            if not self.occupancy.is_free(point):
                print("Path no longer valid...")
                return False
        return True
    
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

    def get_current_plan_time(self):
        # returns the time since the current plan started
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self, obj_x=[], obj_y=[], obj_d=[]):
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
        elif self.mode == Mode.TRACK:
            return  # don't replan if we are already tracking a plan

        current_time = rospy.get_rostime()
 
        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))

        # Get locations of agents who are static
        robots_x = []  # list of x coordinates of other agents who are static
        robots_y = []  # list of y coordinates of other agents who are static
        for agent_id in self.other_agents_static:
            agent_x, agent_y, _, _ = self.other_agents_locations[agent_id]
            robots_x.append(agent_x)
            robots_y.append(agent_y)

        combined_occupancy = self.occupancy

        problem = AStar(state_min, state_max, x_init, x_goal, combined_occupancy, self.plan_resolution,
            robots_x=robots_x, robots_y=robots_y, obj_x=obj_x, obj_y=obj_y, obj_d=obj_d)

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success and self.mode == Mode.IDLE:
            rospy.loginfo("Planning failed")
            self.times_planned_failed += 1

            if self.times_planned_failed > 5:
                rospy.loginfo("Planning failed too many times, stopping")
                self.times_planned_failed = 0

                self.x_g = None
                self.y_g = None
                self.theta_g = None
                self.switch_mode(Mode.IDLE)
            else:
                self.x_g += np.random.normal(0,0.05)
                self.y_g += np.random.normal(0,0.05)
                self.replan()
            return
        else:
            self.times_planned_failed = 0
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

        # Publish the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub, times=t_new)

        # Load the new trajectory into the controllers
        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)
        self.current_plan = traj_new
        self.unsmoothed_plan = planned_path

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)
        
        # Populate the waypoints, use every 20th point:
        self.waypoints = []
        marker_arr = MarkerArray()
        th_init_new = traj_new[0, 2]
        th_err = wrapToPi(th_init_new - self.theta)
        t_init_align = abs(th_err / self.om_max)
        for i in range(20, len(traj_new), 20):
            self.waypoints.append([traj_new[i, 0], traj_new[i, 1], t_new[i]+2])  # +2 to add a buffer +t_init_align+current_time
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = traj_new[i, 0]
            marker.pose.position.y = traj_new[i, 1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_arr.markers.append(marker)
        self.waypoint_pub.publish(marker_arr)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return
        else:
            rospy.loginfo("Ready to track")
            self.switch_mode(Mode.TRACK)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )

            # Check if we need to modify the velocity for a person
            V, om = self.modify_velocity_for_person(V, om)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.BACKING:
            V = -0.3
            om = 0.0
        elif self.mode == Mode.LOCALIZING:
            V = 0.0
            om = 1.5
        elif self.mode == Mode.WAITING_FOR_INIT:
            V = 0.0
            om = 0.0
        else:
            V = 0.0
            om = 0.0

        self.prev_om = om

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)
    
    def run(self):
        """
        Main loop of the navigator node.
        """

        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
        
            # try to get state information to update self.x, self.y, self.theta
            try:
                if self.mode != Mode.LOCALIZING:
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
                if self.received_initial_pose:
                    rospy.loginfo("Navigator: waiting for state info")
                if self.mode != Mode.IDLE and self.mode != Mode.WAITING_FOR_INIT:
                    self.switch_mode(Mode.IDLE)
                    print(e)
                pass

            self.replanning_from_object = False

            # # If not localized, switch to LOCALIZING mode
            # if not self.is_localized and self.mode != Mode.LOCALIZING:
            #     rospy.loginfo("Navigator: not localized, switching to LOCALIZING mode")
            #     self.switch_mode(Mode.LOCALIZING)

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.WAITING_FOR_INIT:
                if self.received_initial_pose:
                    rospy.loginfo("Navigator: Have initial pose, switching to Localize to Get better estimate")
                    self.localize_begin_time = rospy.get_rostime()
                    self.switch_mode(Mode.LOCALIZING)
            elif self.mode == Mode.LOCALIZING:
                # if time spent localizing > 10 sec, switch to ALIGN mode
                if self.localize_begin_time is not None and rospy.get_rostime() - self.localize_begin_time > rospy.Duration(10):
                    rospy.loginfo("Navigator: localized, ready for navigation")
                    self.is_localized = True
                    self.localized_pub.publish(True)
                    self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                current_time = rospy.get_rostime().to_sec()
                if self.near_goal():
                    # We are close to goal ---> Switch to pose controller
                    self.heading_controller.load_goal(self.theta_g)
                    print("Setting theta goal to", self.theta_g)
                    self.switch_mode(Mode.PARK)                
                # elif(not self.path_still_valid(self.current_plan)):
                #     # Path no longer valid ---> replan
                #     rospy.loginfo("replanning because path is no longer valid")

                #     # Stop the robot
                #     self.switch_mode(Mode.IDLE)
                #     cmd_vel = Twist()
                #     cmd_vel.linear.x = 0.0
                #     cmd_vel.angular.z = 0.0
                #     self.nav_vel_pub.publish(cmd_vel)

                #     # Now replan
                #     self.replan()
                elif len(self.waypoints) > 0 and not self.robot_stopped_by_person:
                    # If we have waypoints, check if we have reached them in time
                    if current_time - self.current_plan_start_time.to_sec() > self.waypoints[0][2]:
                        print("******************************************")
                        print("Backing up because haven't reached waypoint")
                        print("******************************************")
                        self.backing_start_time = current_time
                        self.switch_mode(Mode.BACKING)
                    elif np.linalg.norm(np.array([self.x - self.waypoints[0][0], self.y - self.waypoints[0][1]])) < 0.35:
                        print("Waypoint reached")
                        self.waypoints.pop(0)  # Remove the first waypoint since we are close to it
                elif self.robot_stopped_by_person and current_time - self.stopped_for_person_time > 5:
                    # If we have been stopped by a person for more than 5 seconds, replan

                    print("******************************************")
                    print("Replanning because person in path")
                    print("******************************************")

                    # Stop attempting current plan
                    self.switch_mode(Mode.IDLE)

                    self.robot_stopped_by_person = False
                    
                    # Try to replan
                    self.replan()

                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")

                    # Stop attempting current plan
                    self.switch_mode(Mode.IDLE)

                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                # Reached goal: forget goal coordinates and stop
                if self.aligned_goal():
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.BACKING:
                print("Backing Up")
                current_time = rospy.get_rostime().to_sec()
                if current_time - self.backing_start_time > 1:
                    self.switch_mode(Mode.IDLE)
                    
                    # Stop moving
                    cmd_vel = Twist()
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                    self.nav_vel_pub.publish(cmd_vel)

                    # Now replan
                    print("Replanning after backing up")
                    self.replan()

            self.publish_control()
            rate.sleep()

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(3)  # Give time for everything to set up
    nav.run()  # run the main loop