#!/usr/bin/env python3

import rospkg
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped, PoseWithCovarianceStamped, Point
from std_msgs.msg import String, Int32, Float64, Bool
from visualization_msgs.msg import Marker, MarkerArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
from mattbot_dds.msg import AgentPath, AgentLocation
import tf
from dynamic_reconfigure.client import Client

import time
import numpy as np
import networkx as nx
from numpy import linalg
import scipy.interpolate
import matplotlib.pyplot as plt
from enum import Enum
import requests
import os

from navigation_utils import TrajectoryTracker, PoseController, HeadingController, wrapToPi, StochOccupancyGrid2D, AStar, compute_smoothed_traj
from social_path_planning import AStar as SocialAStar, AStar_With_Graph as SocialAStar_With_Graph
from social_path_planning import FrequentSubgraph


V_PREV_THRES = 0.0001

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

PERSON_STOP_DISTANCE = 1.6  # distance to closest person at which we stop the robot
PERSON_SLOW_DISTANCE = 2.5  # distance to closest person at which we slow down the robot
OBJECT_STOP_DISTANCE = 1.5
AGENT_STOP_DISTANCE = 2

class Mode(Enum):
    IDLE = 0        # not moving, waiting for a goal
    LOCALIZING = 1  # localizing the robot
    LOCALIZING2 = 2
    ALIGN = 3       # aligning to start heading of the path
    TRACK = 4       # tracking the path (following the trajectory)
    PARK = 5       # parking the robot (moving to a specific pose)
    BACKING = 6     # backing up when stuck
    WAITING_FOR_INIT = 7
    RELOCALIZING = 8
    STOPPED_FOR_PERSON = 9
    STOPPED_FOR_AGENT = 10

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
        self.object_occupancy = None
        self.frequent = None  # The frequent subgraph
        self.sparse_graph_threshold = rospy.get_param('/sparse_graph_threshold', 5)  # Threshold for building the frequent subgraph
        self.sparse_graph_components = rospy.get_param('/sparse_graph_components', 15)  # Minimum component size for pruning the frequent subgraph
        self.map_frame_id = rospy.get_param('~frequent_graph_map_frame', 'map')
        self.publish_frequent_graph_viz = rospy.get_param('~publish_frequent_graph_viz', True)
        # Applied once when canonicalizing loaded heatmap graph keys to ROS map (col, row).
        self.frequent_graph_mirror_cell_x = rospy.get_param('~frequent_graph_mirror_cell_x', True)
        self.frequent_graph_mirror_cell_y = rospy.get_param('~frequent_graph_mirror_cell_y', True)
        self.frequent_graph_swap_cell_axes = rospy.get_param('~frequent_graph_swap_cell_axes', False)

        self.person_occupancy = None
        self.robot_stopped_by_person = False
        self.person_in_path = False
        self.person_list1 = []
        self.person_list2 = []
        self.person_list3 = []
        self.stopped_for_person_time = rospy.get_rostime().to_sec()

        self.object_near_occupancy = None
        self.robot_stopped_by_object = False
        self.object_in_path = False
        self.object_list1 = []  
        self.object_list2 = []
        self.object_list3 = []  # List to hold detected objects for the last three frames
        self.stopped_for_object_time = rospy.get_rostime().to_sec()

        self.other_agent_locs = dict()
        self.other_agent_time_dict = dict()  # agent ID -> time of last update
        self.agents_in_path = set()  # set of agent ID's that are in the path
        self.other_agents_not_static = []  # list of agent ID's that are not static (moving)
        self.other_agents_static = []  # list of agent ID's that are static (not moving)
        self.my_id = int(os.environ.get('ROBOT_ID', '0'))  # Get the robot ID from environment variable
        self.stopped_for_agents = False
        self.is_lowest_id = False

        self.other_agents_goals = dict()
        self.other_agents_at_goal = []
    
        self.backing_from_bad_localization = False
                                            
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

        self.v_des = rospy.get_param('/cruising_velocity', 0.35) # desired cruising velocity
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
        self.kpx = rospy.get_param('/kpx', 2)
        self.kpy = rospy.get_param('/kpy', 2)
        self.kdx = rospy.get_param('/kdx', 1.5)
        self.kdy = rospy.get_param('/kdy', 1.5)

        # heading controller parameters
        self.kp_th = 1.5
        self.om_prev = 0.0

        # Get AMCL parameters to use
        self.alpha1 = rospy.get_param('/alpha1', 0.8)
        self.alpha2 = rospy.get_param('/alpha2', 0.8)
        self.alpha3 = rospy.get_param('/alpha3', 0.8)
        self.alpha4 = rospy.get_param('/alpha4', 0.8)
        self.z_hit = rospy.get_param('/z_hit', 0.95)
        self.z_rand = rospy.get_param('/z_rand', 0.05)
        self.sigma_hit = rospy.get_param('/sigma_hit', 0.01)

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

        # Get whether to use social A* or Regular A*
        self.use_social_astar = rospy.get_param('/use_social_astar', False)
        if self.use_social_astar:
            rospy.loginfo("Using social A* for path planning")
        else:
            rospy.loginfo("Using regular A* for path planning")

        #Publishes current state of robot (IDLE, ALIGN, etc)
        self.state_pub = rospy.Publisher("/robot_mode", Int32, queue_size=10)

        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.frequent_graph_viz_pub = rospy.Publisher(
            "/frequent_graph_viz", Marker, queue_size=1, latch=True
        )

        self.trans_listener = tf.TransformListener()

        # Get map parameter to determine what map to use
        map_name = rospy.get_param('map_name', '/map')

        # Distance to closest person --- determines if we should slow down or stop
        self.distance_to_person = np.inf

        rospy.Subscriber(map_name, OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/object_map", OccupancyGrid, self.object_map_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber("/external_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/voice_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/agent_location", AgentLocation, self.agent_location_callback)
        rospy.Subscriber("/initialpose_relocalize", PoseWithCovarianceStamped, self.initial_pose_relocalize_callback)
        self.initialpose_subscriber = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_callback)
        rospy.Subscriber("/lost_localization", Bool, self.lost_localization_callback)
        rospy.Subscriber("/detected_objects", DetectedObjectArray, self.detected_objects_callback)
        rospy.Subscriber("/valid_points", Bool, self.valid_points_callback)
        self.localized_pub = rospy.Publisher("/localized", Bool, queue_size=10)
        self.initialpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
        self.invalid_goal_pub = rospy.Publisher("/invalid_goal", Pose2D, queue_size=10)

        self.has_stopped = False

        self.waypoints = []
        self.backing_for_waypoints = False
        self.backing_start_time = 0

        self.switch_mode(Mode.WAITING_FOR_INIT)

        self.url='http://127.0.0.1:5000/gemini'


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

        if self.mode == Mode.WAITING_FOR_INIT or self.mode == Mode.LOCALIZING:
            # Ignore external goals
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
            # Publish an invalid goal message
            invalid_goal_msg = Pose2D()
            invalid_goal_msg.x = msg.x
            invalid_goal_msg.y = msg.y
            invalid_goal_msg.theta = msg.theta
            self.invalid_goal_pub.publish(invalid_goal_msg)
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

    def _heatmap_cell_to_world(self, ci, ri, w, h, ox, oy, res):
        """Heatmap/sparse_graph cell (col, row) → world (m) using load-time mirror/swap params."""
        if self.frequent_graph_swap_cell_axes:
            ci, ri = ri, ci
        if self.frequent_graph_mirror_cell_x:
            ci = w - 1 - ci
        if self.frequent_graph_mirror_cell_y:
            ri = h - 1 - ri
        return ox + float(ci) * res, oy + float(ri) * res

    def _canonical_cell_from_heatmap_indices(self, ci, ri):
        """Map stored heatmap node indices to ROS map cell indices matching occupancy / get_index."""
        occ = self.occupancy
        w, h = int(occ.width), int(occ.height)
        ox, oy = float(occ.origin_x), float(occ.origin_y)
        res = float(occ.resolution)
        wx, wy = self._heatmap_cell_to_world(ci, ri, w, h, ox, oy, res)
        gi = int(np.round((wx - ox) / res))
        gj = int(np.round((wy - oy) / res))
        gi = int(np.clip(gi, 0, w - 1))
        gj = int(np.clip(gj, 0, h - 1))
        return (gi, gj)

    def _canonicalize_frequent_graph(self):
        """Relabel graph nodes so keys are ROS map (col, row); RViz uses origin + index * res."""
        g = self.frequent.graph
        mapping = {}
        collisions = 0
        seen_new = {}
        for node in list(g.nodes()):
            try:
                ci, ri = int(node[0]), int(node[1])
            except (TypeError, ValueError, IndexError):
                rospy.logwarn("Skipping non-integer frequent graph node: %r", node)
                continue
            new_node = self._canonical_cell_from_heatmap_indices(ci, ri)
            mapping[node] = new_node
            if new_node in seen_new and seen_new[new_node] != node:
                collisions += 1
            seen_new.setdefault(new_node, node)
        if not mapping:
            return
        # copy=True: mapping often overlaps old/new labels (e.g. identity on many nodes);
        # in-place relabel then raises NetworkXUnfeasible (cycle / overlapping label sets).
        self.frequent.graph = nx.relabel_nodes(g, mapping, copy=True)
        g = self.frequent.graph
        rospy.loginfo(
            "Canonicalized frequent subgraph to ROS map cells: %d nodes, %d edges%s",
            g.number_of_nodes(),
            g.number_of_edges(),
            (" (%d heatmap cells merged to same ROS cell)" % collisions) if collisions else "",
        )

    def publish_frequent_graph_rviz(self):
        """Publish the loaded frequent subgraph as a LINE_LIST Marker in ``map_frame_id``."""
        if not self.publish_frequent_graph_viz:
            return
        if self.frequent is None or self.occupancy is None:
            return
        g = self.frequent.graph
        if g.number_of_edges() == 0:
            rospy.logwarn("Frequent graph has no edges; skipping RViz visualization.")
            return

        m = Marker()
        m.header.frame_id = self.map_frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "frequent_subgraph"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = rospy.get_param('~frequent_graph_line_width', 0.015)
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = rospy.get_param('~frequent_graph_alpha', 0.65)
        m.lifetime = rospy.Duration(0)

        ox = float(self.occupancy.origin_x)
        oy = float(self.occupancy.origin_y)
        res = float(self.occupancy.resolution)
        z = float(rospy.get_param('~frequent_graph_z', 0.02))

        def _cell_to_world(ci, ri):
            """Nodes are canonical ROS map (col, row) after _canonicalize_frequent_graph."""
            return ox + float(ci) * res, oy + float(ri) * res

        for u, v in g.edges():
            try:
                uc, ur = int(u[0]), int(u[1])
                vc, vr = int(v[0]), int(v[1])
            except (TypeError, ValueError, IndexError):
                continue
            x0, y0 = _cell_to_world(uc, ur)
            x1, y1 = _cell_to_world(vc, vr)
            m.points.append(Point(x=x0, y=y0, z=z))
            m.points.append(Point(x=x1, y=y1, z=z))

        self.frequent_graph_viz_pub.publish(m)
        rospy.loginfo(
            "Published /frequent_graph_viz: %d edges (%d line vertices) frame=%s",
            g.number_of_edges(),
            len(m.points),
            self.map_frame_id,
        )

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        if msg.header.frame_id:
            self.map_frame_id = msg.header.frame_id
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:

        # FIXME: We need to update the occupancy grid map every time we get a new message, but for now just get the single map message
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
            and self.occupancy is None
        ):

            print("+"*50)
            print("Assigned Occupancy Grid Map!")
            print("+"*50)

            
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('path_planning')
            wall_distance_cache = pkg_path + '/src/social_path_planning/environments/y2e2_aligned_wall_dist.npz'
            
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,
                self.map_probs,
                wall_distance_cache_path=wall_distance_cache
            )

            if self.frequent is None:  # Don't load the graph every time
                # Get Heat Map File Name Prefix
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path('path_planning')
                heatmap_prefix = pkg_path + '/ros_map'

                # Check if heatmap file actually exists before trying to load it
                heatmap_name = heatmap_prefix + '_heatmap.npy'
                if os.path.isfile(heatmap_name):
                    self.frequent = FrequentSubgraph(self.occupancy, heat_map_filename=heatmap_prefix)
                    self.frequent.build_graph(threshold=self.sparse_graph_threshold, reset_graph=True)
                    rospy.logwarn(f"Number of nodes/edges in graph before pruning = {len(self.frequent.graph.nodes)}/{len(self.frequent.graph.edges)}")
                    self.frequent.prune_graph(min_component_size=self.sparse_graph_components)
                    rospy.logwarn(f"Number of nodes/edges in graph = {len(self.frequent.graph.nodes)}/{len(self.frequent.graph.edges)}")
                    self._canonicalize_frequent_graph()
                    self.publish_frequent_graph_rviz()
                else:
                    rospy.logwarn(f"Heatmap file {heatmap_name} not found, skipping loading frequent subgraph.")
            
            if self.object_occupancy is None:
                self.object_occupancy = StochOccupancyGrid2D(
                    self.map_resolution,
                    self.map_width,
                    self.map_height,
                    self.map_origin[0],
                    self.map_origin[1],
                    5,
                    np.zeros((self.map_width * self.map_height,)),  # initialize with zeros
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

                self.object_near_occupancy = StochOccupancyGrid2D(
                    self.map_resolution,
                    self.map_width,
                    self.map_height,
                    self.map_origin[0],
                    self.map_origin[1],
                    5,
                    np.zeros((self.map_width * self.map_height,)),  # initialize with zeros     
                )

                

    def object_map_callback(self, msg):
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            print()

            self.object_occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,
                msg.data,
            )

    def initial_pose_relocalize_callback(self, msg):
        """
        Callback for initial pose, sets the robot's position and orientation
        """        
        # Only do anything if in tracking mode
        if self.mode == Mode.TRACK:
            self.initialpose_pub.publish(msg)
            self.backing_start_time = rospy.get_rostime().to_sec()
            self.backing_from_bad_localization = True
            self.switch_mode(Mode.BACKING)

    def initial_pose_callback(self, msg):
        """
        Callback for initial pose, sets the robot's position and orientation
        """        
        rospy.loginfo("Navigator: Initial pose received")
        self.received_initial_pose = True

        if self.initialpose_subscriber is not None:
            self.initialpose_subscriber.unregister()
            self.initialpose_subscriber = None


    def lost_localization_callback(self, msg):
        self.backing_start_time = rospy.get_rostime().to_sec()
        self.switch_mode(Mode.BACKING)


        # if self.localized and msg.data:
        #     rospy.loginfo("Navigator: Lost localization")
        #     self.localized = False
        #     self.received_initial_pose = False
        #     self.switch_mode(Mode.WAITING_FOR_INIT)

    def agent_location_callback(self, msg):
        """
        Callback for /agent_location topic.
        Updates the location of other agents and checks if they are static.
        """
        current_time = rospy.get_rostime()
        agent_id = msg.agentID.data

        self.other_agent_time_dict[agent_id] = current_time.to_sec()
        self.other_agent_locs[agent_id] = (msg.pose.position.x, msg.pose.position.y)

        # Check if static and update the static list
        if agent_id not in self.other_agents_static and msg.isStatic.data:
            self.other_agents_static.append(agent_id)
        elif agent_id in self.other_agents_static and not msg.isStatic.data:
            self.other_agents_static.remove(agent_id)

        if agent_id in self.other_agents_not_static and msg.isStatic.data:
            self.other_agents_not_static.remove(agent_id)
        elif agent_id not in self.other_agents_not_static and not msg.isStatic.data:
            self.other_agents_not_static.append(agent_id)

    def valid_points_callback(self, msg):
        if not msg.data:
            rospy.logwarn("Invalid points received, stopping the robot.")

            if self.mode == Mode.TRACK:
                # For now, use backing for waypoints
                current_time = rospy.get_rostime().to_sec()
                self.backing_start_time = current_time
                self.backing_for_waypoints = True
                self.switch_mode(Mode.BACKING)

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
        try:
            x0 = self.plan_resolution * round(x[0] / self.plan_resolution)
            x1 = self.plan_resolution * round(x[1] / self.plan_resolution)
            return (x0, x1)
        except Exception as e:
            rospy.logerr("Error snapping to grid: %s", e)
            return x

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

    def object_intersect_path(self):
        object_probs = self.object_near_occupancy.get_probs()
        path = self.current_plan
        for point in path:
            grid_x = int((point[0] - self.map_origin[0]) / self.map_resolution)
            grid_y = int((point[1] - self.map_origin[1]) / self.map_resolution)
            if (0 <= grid_x < self.map_width) and (0 <= grid_y < self.map_height):
                if object_probs[grid_y, grid_x] > 0.5:
                    return True
        return False

    def agent_intersect_path(self):
        # First remove any invalid agents (time too long ago)
        current_time = rospy.get_rostime().to_sec()
        invalid_agents = []
        close_agents = []
        for agent_id in list(self.other_agent_locs.keys()):
            if (current_time - self.other_agent_time_dict[agent_id]) > 5.0:
                invalid_agents.append(agent_id)
            else:
                agent_x, agent_y = self.other_agent_locs[agent_id]
                dist_to_agent = np.linalg.norm(np.array([self.x - agent_x, self.y - agent_y]))
                if dist_to_agent < AGENT_STOP_DISTANCE:
                    close_agents.append(agent_id)

        # Remove invalid agents from the dictionaries
        for agent_id in invalid_agents:
            self.other_agent_locs.pop(agent_id)
            self.other_agent_time_dict.pop(agent_id)
       
        # Check if any of the close agents are in the path
        path = self.current_plan
        agents_in_path = set()
        for point in path:
            point_x = point[0]
            point_y = point[1]
            for agent_id in close_agents:
                agent_x, agent_y, = self.other_agent_locs[agent_id]
                grid_x = int((point_x - self.map_origin[0]) / self.map_resolution)
                grid_y = int((point_y - self.map_origin[1]) / self.map_resolution)

                dist_to_agent = np.linalg.norm(np.array([point_x - agent_x, point_y - agent_y]))
                if dist_to_agent < 0.45:
                    agents_in_path.add(agent_id)

        self.agents_in_path = agents_in_path
        
        # Reset is_lowest_id if there are no agents in the path
        if len(self.agents_in_path) == 0:
            self.is_lowest_id = False

        return len(agents_in_path) > 0  # Return True if any agents are in the path

    def detected_objects_callback(self, msg):
        """
        receives detected objects, only looks at people and stores their location in the 
        self.person_occupancy occupancy grid map
        """

        # Each detection stays valid for 3 frames
        self.person_list3 = self.person_list2  # 2 frames ago
        self.person_list2 = self.person_list1
        self.person_list1 = []  # Most recent --- now

        self.object_list3 = self.object_list2  # 2 frames ago
        self.object_list2 = self.object_list1
        self.object_list1 = []  # Most recent --- now

        # Get the distance to the closest person
        closest_person_dist = np.inf
        person_probs = np.zeros((self.map_height, self.map_width))

        closest_object_dist = np.inf
        object_probs = np.zeros((self.map_height, self.map_width))

        for obj in msg.objects:
            x = obj.pose.position.x
            y = obj.pose.position.y
            width = obj.width

            if obj.class_name == "person":
                self.person_list1.append((x, y))
            else:
                self.object_list1.append((x, y, width))

        if self.person_occupancy is not None:

            for (x,y) in self.person_list1 + self.person_list2 + self.person_list3: 
                radius = 0.3  # assume person occupies a circle of radius 0.3m

                # Get x,y coordinates in terms of grid coordinates
                grid_x = int((x - self.map_origin[0]) / self.map_resolution)
                grid_y = int((y - self.map_origin[1]) / self.map_resolution)

                # make sure coordinates are within map
                grid_x = np.clip(grid_x, 0, self.map_width - 1)
                grid_y = np.clip(grid_y, 0, self.map_height - 1)

                # Update the person occupancy grid
                # Create a grid of coordinates centered at the person
                window_size = self.person_occupancy.window_size
                i_coords, j_coords = np.meshgrid(
                    np.arange(-window_size//2, window_size//2 + 1),
                    np.arange(-window_size//2, window_size//2 + 1)
                )
                
                # Calculate the grid positions
                grid_x_coords = grid_x + i_coords
                grid_y_coords = grid_y + j_coords
                
                # Calculate distances from center (person position)
                distances = np.sqrt(i_coords**2 + j_coords**2) * self.map_resolution
                
                # Create a mask for valid positions (within map bounds and within radius)
                valid_mask = (
                    (grid_x_coords >= 0) & (grid_x_coords < self.map_width) &
                    (grid_y_coords >= 0) & (grid_y_coords < self.map_height) &
                    (distances <= radius)
                )
                
                # Update the occupancy grid for valid positions
                person_probs[grid_y_coords[valid_mask], grid_x_coords[valid_mask]] = 1.0

                dist_to_person = np.linalg.norm(np.array([x - self.x, y - self.y]))

                if dist_to_person < closest_person_dist:
                    closest_person_dist = dist_to_person
            self.distance_to_person = closest_person_dist
            self.person_occupancy.update(person_probs)
            self.person_in_path = self.person_intersect_path()

        if self.object_near_occupancy is not None:
            for (x,y,w) in self.object_list1 + self.object_list2 + self.object_list3:
                radius = w / 2.0

                # Get x,y coordinates in terms of grid coordinates
                grid_x = int((x - self.map_origin[0]) / self.map_resolution)
                grid_y = int((y - self.map_origin[1]) / self.map_resolution)

                # make sure coordinates are within map
                grid_x = np.clip(grid_x, 0, self.map_width - 1)
                grid_y = np.clip(grid_y, 0, self.map_height - 1)

                # Update the object near occupancy grid
                # Create a grid of coordinates centered at the object
                window_size = self.object_near_occupancy.window_size
                i_coords, j_coords = np.meshgrid(
                    np.arange(-window_size//2, window_size//2 + 1),
                    np.arange(-window_size//2, window_size//2 + 1)
                )
                
                # Calculate the grid positions
                grid_x_coords = grid_x + i_coords
                grid_y_coords = grid_y + j_coords

                # Calculate distances from center (object position)
                distances = np.sqrt(i_coords**2 + j_coords**2) * self.map_resolution
                
                # Create a mask for valid positions (within map bounds and within radius)
                valid_mask = (
                    (grid_x_coords >= 0) & (grid_x_coords < self.map_width) &
                    (grid_y_coords >= 0) & (grid_y_coords < self.map_height) &
                    (distances <= radius)
                )
                
                # Update the occupancy grid for valid positions
                object_probs[grid_y_coords[valid_mask], grid_x_coords[valid_mask]] = 1.0

                dist_to_object = np.linalg.norm(np.array([x - self.x, y - self.y]))
                if dist_to_object < closest_object_dist:
                    closest_object_dist = dist_to_object

            self.distance_to_object = closest_object_dist
            self.object_near_occupancy.update(object_probs)
            self.object_in_path = self.object_intersect_path()

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

                data = {'query': "Excuse Me!", 'query_type': 'print_to_screen'}
                try:
                    response = requests.post(self.url, json=data)
                except requests.exceptions.RequestException as e:
                    print(f"Error sending request: {e}")

                self.switch_mode(Mode.STOPPED_FOR_PERSON)

            self.robot_stopped_by_person = True
        elif self.distance_to_person < PERSON_SLOW_DISTANCE:
            print("Slowing down for person")
            # Slow down
            V *= 0.5
            om *= 0.5
        elif self.robot_stopped_by_person:
            # If we were stopped by a person, we can start moving again
            self.robot_stopped_by_person = False

            print("Resuming motion after stopping for person")

            self.replan()

        return V, om

    def replan_for_object(self):
        """
        Modifies the velocity based on the distance to the closest object.
        If the distance is less than a threshold, it stops the robot.
        """
        if self.distance_to_object < OBJECT_STOP_DISTANCE and self.object_in_path:
            return True
        else:
            return False
    
    def path_still_valid(self, path):
        for point in path:
            if not self.object_occupancy.is_free(point):
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
            agent_x, agent_y = self.other_agent_locs[agent_id]
            robots_x.append(agent_x)
            robots_y.append(agent_y)

        # Get agents who are in path and avoid them
        for agent_id in self.agents_in_path:
            agent_x, agent_y = self.other_agent_locs[agent_id]
            robots_x.append(agent_x)
            robots_y.append(agent_y)

        combined_occupancy = self.occupancy

        if self.use_social_astar:
            # First determine if a social graph is available to use, if not, fall back to regular social A*
            if self.frequent is not None:
                # Graph nodes are occupancy map cells: get_index must use occ.resolution (same as heat_map / sim).
                # plan_resolution is coarser and breaks has_edge matching under ROS. If planning is too slow at
                # map resolution, consider downsampling the heatmap/graph to plan_resolution (larger change).
                graph_planner_resolution = float(combined_occupancy.resolution)
                problem = SocialAStar_With_Graph(
                    state_min,
                    state_max,
                    x_init,
                    x_goal,
                    combined_occupancy,
                    self.frequent.graph,
                    resolution=graph_planner_resolution,
                    desired_dist_right_extra=0.25,
                )

                print("+"*5 + "Using social A* with graph" + "+"*5)
            else:
                problem = SocialAStar(state_min, state_max, x_init, x_goal, combined_occupancy, self.plan_resolution)
                print("+"*5 + "Using social A*" + "+"*5)
        else:
            problem = AStar(state_min, state_max, x_init, x_goal, combined_occupancy, self.plan_resolution,
            robots_x=robots_x, robots_y=robots_y, obj_x=obj_x, obj_y=obj_y, obj_d=obj_d)

        rospy.loginfo("Navigator: computing navigation plan")
        success = False
        while not success:
            success = problem.solve()
            if not success and (self.mode == Mode.IDLE or self.mode == Mode.STOPPED_FOR_AGENT):
                rospy.loginfo("Planning failed")
                self.times_planned_failed += 1

                if self.times_planned_failed > 1:
                    rospy.loginfo("Planning failed too many times, stopping")
                    self.times_planned_failed = 0

                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
                    return
            else:
                self.times_planned_failed = 0
                rospy.loginfo("Planning Succeeded")
                planned_path = problem.path

        if self.use_social_astar and self.frequent is not None:
            tel = getattr(problem, "last_solve_telemetry", None)
            if isinstance(tel, dict):
                rospy.loginfo(
                    "social A* with graph: path_fraction_on_graph=%s on_graph=%s/%s "
                    "graph_edge_cost_evals=%s off_graph_social_evals=%s",
                    tel.get("path_fraction_on_graph"),
                    tel.get("path_segments_on_graph"),
                    tel.get("path_segments_total"),
                    tel.get("graph_edge_cost_evals"),
                    tel.get("off_graph_social_evals"),
                )

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
        elif self.mode == Mode.LOCALIZING or self.mode == Mode.RELOCALIZING:
            V = 0.0
            om = 1.5
        elif self.mode == Mode.LOCALIZING2:
            if self.localize_begin_time is not None and rospy.get_rostime() - self.localize_begin_time < rospy.Duration(2):
                # If we are still localizing, use a high angular velocity to align
                V = 0.0
                om = 0.0
            else:
                V = 0.0
                om = -1.5
        elif self.mode == Mode.WAITING_FOR_INIT:
            V = 0.0
            om = 0.0
        elif self.mode == Mode.STOPPED_FOR_PERSON:
            # If we are stopped for a person, we don't want to move
            V = 0.0
            om = 0.0
        elif self.mode == Mode.STOPPED_FOR_AGENT:
            # If we are stopped for an agent, we don't want to move
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
                    self.localize_begin_time = rospy.get_rostime()
                    
                    self.switch_mode(Mode.LOCALIZING2)
            elif self.mode == Mode.LOCALIZING2:
                # if time spent localizing > 5 sec, switch to ALIGN mode
                if self.localize_begin_time is not None and rospy.get_rostime() - self.localize_begin_time > rospy.Duration(5):
                    rospy.loginfo("Navigator: localized, ready for navigation")
                    self.is_localized = True
                    self.localized_pub.publish(True)
                    self.switch_mode(Mode.IDLE)

                    client = Client('amcl', timeout=30)
                    # Get current configuration
                    config = client.get_configuration()
                    # rospy.loginfo("Current configuration: %s", config)

                    # Update parameters to emphasize sensor measurements
                    params = {
                        # Measurement model parameters
                        'laser_z_hit': self.z_hit,            # Increase hit weight (default ~0.7)
                        'laser_z_rand': self.z_rand,          # Decrease random weight (default ~0.2)
                        'laser_sigma_hit': self.sigma_hit,        # Decrease sigma for higher confidence
                        'odom_alpha1': self.alpha1,
                        'odom_alpha2': self.alpha2,
                        'odom_alpha3': self.alpha3,
                        'odom_alpha4': self.alpha4
                    }
                    client.update_configuration(params)
                    print("Updated AMCL parameters for better localization after initial pose.")

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
                elif(not self.path_still_valid(self.current_plan)):
                    # Path no longer valid ---> replan
                    rospy.loginfo("replanning because path is no longer valid")

                    # Stop the robot
                    self.switch_mode(Mode.IDLE)
                    cmd_vel = Twist()
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                    self.nav_vel_pub.publish(cmd_vel)

                    # Now replan
                    self.replan()
                elif len(self.waypoints) > 0 and not self.robot_stopped_by_person:
                    # If we have waypoints, check if we have reached them in time
                    if current_time - self.current_plan_start_time.to_sec() > self.waypoints[0][2]:
                        print("******************************************")
                        print("Backing up because haven't reached waypoint")
                        print("******************************************")
                        self.backing_start_time = current_time
                        self.backing_for_waypoints = True
                        self.switch_mode(Mode.BACKING)
                    elif np.linalg.norm(np.array([self.x - self.waypoints[0][0], self.y - self.waypoints[0][1]])) < 0.35:
                        print("Waypoint reached")
                        self.waypoints.pop(0)  # Remove the first waypoint since we are close to it
                
                if self.agent_intersect_path() and not self.is_lowest_id:
                    self.switch_mode(Mode.STOPPED_FOR_AGENT)
                    print("Agent in Path---Stopping")
                    self.stopped_for_agents = True

                # if self.replan_for_object():
                #     # If we are too close to an object, stop and replan
                #     print("******************************************")
                #     print("Replanning because object in path")
                #     print("******************************************")

                #     self.replanning_from_object = True

                #     # Stop the robot
                #     self.switch_mode(Mode.IDLE)
                #     cmd_vel = Twist()
                #     cmd_vel.linear.x = 0.0
                #     cmd_vel.angular.z = 0.0
                #     self.nav_vel_pub.publish(cmd_vel)

                #     obj_x = []  
                #     obj_y = []
                #     obj_d = []
                #     for (x, y, d) in self.object_list1 + self.object_list2 + self.object_list3:
                #         obj_x.append(x)
                #         obj_y.append(y)
                #         obj_d.append(d)

                #     # Now replan
                #     self.replan(obj_x=obj_x, obj_y=obj_y, obj_d=obj_d)

                if (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
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

                    if False:  # self.backing_from_bad_localization:
                        print("Backing up from bad localization")
                        # Stop moving
                        cmd_vel = Twist()
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        self.nav_vel_pub.publish(cmd_vel)

                        self.backing_from_bad_localization = False
                        self.relocalizing_start_time = current_time
                        self.switch_mode(Mode.RELOCALIZING)
                    elif self.backing_for_waypoints:
                        print("Backing up for waypoints")
                        # Stop moving
                        cmd_vel = Twist()
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        self.nav_vel_pub.publish(cmd_vel)

                        self.backing_for_waypoints = False

                        # Now relocalize
                        self.relocalizing_start_time = current_time
                        self.switch_mode(Mode.RELOCALIZING)
                    else:
                        self.switch_mode(Mode.IDLE)

                        # Stop moving
                        cmd_vel = Twist()
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        self.nav_vel_pub.publish(cmd_vel)

                        # Now replan
                        print("Replanning after backing up")
                        self.replan()

                    # self.relocalizing_start_time = current_time
                    # self.switch_mode(Mode.RELOCALIZING)
            elif self.mode == Mode.RELOCALIZING:
                current_time = rospy.get_rostime().to_sec()
                if current_time - self.relocalizing_start_time > 5:
                    self.switch_mode(Mode.IDLE)
                    
                    # Stop moving
                    cmd_vel = Twist()
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                    self.nav_vel_pub.publish(cmd_vel)

                    self.replan()
            elif self.mode == Mode.STOPPED_FOR_PERSON:
                current_time = rospy.get_rostime().to_sec()
                if current_time - self.stopped_for_person_time > 5:
                    # If we have been stopped by a person for more than 5 seconds, replan
                    print("******************************************")
                    print("Replanning because person in path")
                    print("******************************************")

                    self.switch_mode(Mode.IDLE)
                    self.robot_stopped_by_person = False
                    obj_x = []
                    obj_y = []
                    obj_d = []
                    for (x, y) in self.person_list1 + self.person_list2 + self.person_list3:
                        obj_x.append(x)
                        obj_y.append(y)
                        obj_d.append(0.3)  # Assuming a diameter of 0.3m for people

                    self.replan(obj_x=obj_x, obj_y=obj_y, obj_d=obj_d)
                elif self.distance_to_person > PERSON_STOP_DISTANCE:
                    print("******************************************")
                    print("Replanning because person no longer in path")
                    print("******************************************")
                    
                    self.switch_mode(Mode.IDLE)
                    self.robot_stopped_by_person = False
                    self.replan()

            elif self.mode == Mode.STOPPED_FOR_AGENT:

                if not self.agent_intersect_path():
                    # If there are no agents in the path, we can replan
                    print("******************************************")
                    print("Replanning because agent no longer in path")
                    print("******************************************")

                    self.switch_mode(Mode.IDLE)
                    self.replan()
                else:
                    # Get intersection of agents in path and agents that are not static
                    moving_agents_in_path = self.agents_in_path.intersection(set(self.other_agents_not_static))
                    if len(moving_agents_in_path) == 0:
                        # No moving agents, replan around the static agents:
                        print("******************************************")
                        print("Replanning because no moving agents in path")
                        print("******************************************")
                        print("Agents in path:", self.agents_in_path)
                        print("other agents not static:", self.other_agents_not_static)                        
                        self.replan()
                    else:
                        # We have moving agents in the path, the smallest agent_id has priority
                        # and we will wait for it to move
                        lowest_agent_id = min(moving_agents_in_path)
                        if self.my_id <= lowest_agent_id:
                            # If we are the lowest agent, we can replan
                            print("******************************************")
                            print("Replanning because we are the lowest agent in path")
                            print("******************************************")
                            self.is_lowest_id = True
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