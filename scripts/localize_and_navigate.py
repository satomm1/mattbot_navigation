#!/usr/bin/env python3

import time
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String, Int32, Float64, Bool
from visualization_msgs.msg import Marker, MarkerArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
from mattbot_dds.msg import AgentPath, AgentLocation
import tf
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

class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    BACKING = 4       
   

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

        self.other_agents_paths = dict()
        self.other_agents_goals = dict()
        self.other_agents_at_goal = []
        self.other_agents_static = []
        self.other_agents_locations = dict()
                                            
        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 500

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.7  # maximum velocity
        self.om_max = 3  # maximum angular velocity
        self.om_heading = 1.3  # angular velocity for heading controller

        self.v_des = 0.5  # desired cruising velocity
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
        # map_name = "/map"

        rospy.Subscriber(map_name, OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber("/external_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/voice_goal", Pose2D, self.external_goal_callback)
        rospy.Subscriber("/path_from_agent", AgentPath, self.path_from_agent_callback)
        rospy.Subscriber("/agent_location", AgentLocation, self.agent_location_callback)
        # rospy.Subscriber("/detected_objects", DetectedObjectArray, self.detected_objects_callback)
        self.localized_sub = rospy.Subscriber("/localized", Bool, self.localized_callback)

        self.has_stopped = False

        self.waypoints = []
        self.backing_start_time = 0

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

        if self.occupancy is not None and not self.occupancy.is_free((x_g_proposed, y_g_proposed)):
            rospy.loginfo("Not a valid goal")
            return
        
        self.x_g = x_g_proposed
        self.y_g = y_g_proposed
        self.theta_g = msg.theta
        self.replan()

    def detected_objects_callback(self, msg):
        """
        receives detected objects and updates the map
        """
        # self.detected_objects = []

        num_existing_objects = len(self.detected_objects)
        indx_matching_objects = []

        object_array = msg.objects
        for obj in object_array:
            x = obj.pose.position.x
            y = obj.pose.position.y
            w = obj.width

            object_already_exists = False
            for i in range(len(self.detected_objects)):
                if np.linalg.norm(np.array([x - self.detected_objects[i][0], y - self.detected_objects[i][1]]) < 0.2):
                    self.detected_objects[i] = (x, y, w, 0)
                    object_already_exists = True
                    indx_matching_objects.append(i)
                    break
            
            if not object_already_exists:
                self.detected_objects.append((x, y, w, 0))

        # Increment the counter for objects that were not detected
        for i in range(num_existing_objects):
            if i not in indx_matching_objects:
                self.detected_objects[i] = (self.detected_objects[i][0], self.detected_objects[i][1], self.detected_objects[i][2], self.detected_objects[i][3] + 1)
        
        # Remove objects that were not detected for a certain number of frames
        self.detected_objects = [obj for obj in self.detected_objects if obj[3] < 60]

            
            # Add to list if doesn't already exist
            # if not any(
            #     np.linalg.norm(np.array([x - existing_obj[0], y - existing_obj[1]]) < 0.2)
            #     for existing_obj in self.detected_objects
            # ):
            #     self.detected_objects.append((x, y, w, 0))


    def localized_callback(self, msg):
        self.is_localized = msg.data

    def path_from_agent_callback(self, msg):
        """
        receives path from agent and updates the map
        """
        agent_id = msg.agentID.data
        path = msg.path
        goal = [path.poses[-1].pose.position.x, path.poses[-1].pose.position.y]
        self.other_agents_paths[agent_id] = path
        self.other_agents_goals[agent_id] = goal

        # Agent not at goal
        if agent_id in self.other_agents_at_goal:
            self.other_agents_at_goal.remove(agent_id)

        print("received new path from agent")

    def agent_location_callback(self, msg):
        

        current_time = rospy.get_rostime()
        agent_id = msg.agentID.data

        prev_pose = None
        if agent_id in self.other_agents_locations:
            prev_pose = self.other_agents_locations[agent_id]

        agent_pose = msg.pose
        x = agent_pose.position.x
        y = agent_pose.position.y
        quaternion = (agent_pose.orientation.x, agent_pose.orientation.y, agent_pose.orientation.z, agent_pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        theta = euler[2]
        self.other_agents_locations[agent_id] = [x, y, theta, current_time]

        if prev_pose is not None:
            if np.linalg.norm(np.array([x - prev_pose[0], y - prev_pose[1]]) < 0.1) and np.abs(theta - prev_pose[2]) < 0.1:
                if agent_id not in self.other_agents_static:
                    self.other_agents_static.append(agent_id)
            else: 
                if agent_id in self.other_agents_static:
                    self.other_agents_static.remove(agent_id)

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

    def path_intersects_obstacle(self, path):
        for point in path:
            for obj in self.detected_objects:
                x_obj, y_obj, obj_diameter, _ = obj
                distance = np.sqrt((point[0] - x_obj)**2 + (point[1] - y_obj)**2)
                if distance <= obj_diameter/2 + 0.3: # 0.3 is the radius of the robot
                    obj_x = []
                    obj_y = []
                    obj_d = []
                    for obj in self.detected_objects:
                        obj_x.append(obj[0])
                        obj_y.append(obj[1])
                        obj_d.append(obj[2])
                    return True, obj_x, obj_y, obj_d
        return False, [], [], []

    def path_still_valid(self, path):
        for point in path:
            if not self.occupancy.is_free(point):
                print("Path no longer valid...")
                print(point)
                return False
        return True

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
        elif self.mode == Mode.BACKING:
            V = -0.4
            om = 0.0
        else:
            V = 0.0
            om = 0.0

        self.prev_om = om

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

    def paths_intersect(self, planned_path, planned_times, other_path):
        current_time = rospy.get_rostime()
        for pose in other_path.poses:
            other_time = pose.header.stamp

            # Time already passed
            if other_time < current_time:
                continue

            for ii in range(len(planned_path)):
                planned_time = planned_times[ii] + current_time
                
                # Times are close enough
                if np.abs(planned_time.to_sec() - other_time.to_sec()) < 0.5:
                    point = planned_path[ii]
                    distance = np.sqrt((point[0] - pose.pose.position.x)**2 + (point[1] - pose.pose.position.y)**2)

                    # Agents are close enough
                    if distance < 0.5:
                        return True
                    
        return False

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

        current_time = rospy.get_rostime()
        # Remove old paths
        agents_to_remove = []
        for agent_id, path in self.other_agents_paths.items():
            if len(path.poses) == 0:
                continue
            
            goal_time = path.poses[-1].header.stamp
            if goal_time < current_time:
                agents_to_remove.append(agent_id)
                continue
        for agent_id in agents_to_remove:
            del self.other_agents_paths[agent_id]

        # # Check if other robot goals are close to ours
        # for agent_id, goal in self.other_agents_goals.items():
        #     if np.linalg.norm(np.array([self.x_g - goal[0], self.y_g - goal[1]]) < 0.5):
        #         # rospy.loginfo("Other agent is close to our goal. Waiting for them to move.")
        #         # self.switch_mode(Mode.IDLE)
        #         continue
        #         # TODO: 

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        # robots_x, robots_y, robots_theta = self.get_stopped_robot_locations()
        robots_x = []
        robots_y = []
        for agent_id in self.other_agents_static:
            agent_x, agent_y, agent_theta, _ = self.other_agents_locations[agent_id]
            robots_x.append(agent_x)
            robots_y.append(agent_y)
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
            robots_x=robots_x,
            robots_y=robots_y,
            obj_x=obj_x,
            obj_y=obj_y,
            obj_d=obj_d
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

            if self.replanning_from_object:
                self.publish_planned_path(planned_path, self.nav_planned_path_pub)
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub, times=t_new)

                self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
                self.traj_controller.load_traj(t_new, traj_new)
                self.current_plan = traj_new
                self.unsmoothed_plan = planned_path

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
            elif t_remaining_new > t_remaining_curr:
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
        current_time = rospy.get_rostime().to_sec()
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
            cmd_vel.angular.z = 1.5
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

            self.replanning_from_object = False

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                path_blocked, obj_x, obj_y, obj_d = self.path_intersects_obstacle(self.current_plan)
                current_time = rospy.get_rostime().to_sec()
                if self.near_goal():
                    self.heading_controller.load_goal(self.theta_g)
                    print("Setting theta goal to", self.theta_g)
                    self.switch_mode(Mode.PARK)
                # elif not self.close_to_plan_start():
                #     rospy.loginfo("replanning because far from start")
                #     self.replan()
                
                elif(path_blocked):
                    rospy.loginfo("replanning because path intersects obstacle")
                    # set controls to zero
                    self.replanning_from_object = True
                    self.replan(obj_x=obj_x, obj_y=obj_y, obj_d=obj_d)
                elif(not self.path_still_valid(self.unsmoothed_plan)):
                    rospy.loginfo("replanning because path is no longer valid")
                    # self.replanning_from_object = True
                    # self.replan()

                    self.switch_mode(Mode.IDLE)
                    # Stop the robot
                    cmd_vel = Twist()
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                    self.nav_vel_pub.publish(cmd_vel)

                    # Now replan
                    self.replan()
                elif len(self.waypoints) > 0:
                    if current_time - self.current_plan_start_time.to_sec() > self.waypoints[0][2]:
                        print("******************************************")
                        print("Backing up because haven't reached waypoint")
                        print("******************************************")
                        # self.replan()
                        self.backing_start_time = current_time
                        self.switch_mode(Mode.BACKING)
                    elif np.linalg.norm(np.array([self.x - self.waypoints[0][0], self.y - self.waypoints[0][1]])) < 0.35:
                        print("Waypoint reached")
                        self.waypoints.pop(0)  # Remove the first waypoint since we are close to it
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
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
            # time.sleep(0.01)


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(5)  # Give time for everything to set up
    nav.localize()  # rotate to localize
    nav.run()  # run the main loop
