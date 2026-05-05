#!/usr/bin/env python3
"""
Navigator variant: same behavior as localize_and_navigate2, plus /external_goal_multi
(MultiRobotExternalGoal). After a successful plan from a multi-robot goal, enters
MULTIAGENT_CONTROL_COMPUTING (integer 11 on /robot_mode), publishes the ego planned path
for DDS, waits for peer paths (fleet_robot_ids + plan_id) or times out, then runs
MultiAgentSimultaneousPlanner in a background thread, builds a timed spline trajectory,
waits for MultiAgentExecuteAt (DDS or auto from fleet leader), then starts TRACK/ALIGN with synchronized t=0 at execute_at.

Python 3.8 does not allow subclassing an existing Enum with new members, so the extra
state uses a separate IntEnum with value 11 (distinct from localize_and_navigate2.Mode).
"""

import importlib.util
import os
import threading
import time
from enum import IntEnum

import numpy as np
import rospy
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Path
from mattbot_dds.msg import MultiAgentExecuteAt, MultiAgentPlannedPath, MultiRobotExternalGoal
from navigation_utils import compute_trajectory_from_timed_waypoints
from social_path_planning.occupancy_grid import StochOccupancyGrid2D as SpStochOccupancyGrid2D
from visualization_msgs.msg import Marker, MarkerArray
from social_path_planning.multi_planning import MultiAgentSimultaneousPlanner


def _load_localize_and_navigate2():
    """
    Load the navigator implementation from scripts/localize_and_navigate2.py.

    We cannot ``import localize_and_navigate2`` when this file is run via the Catkin
    devel wrapper: that name resolves to another wrapper script in lib/pkg/ which execs
    the real file into a private dict, so the imported module has no ``Navigator``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "localize_and_navigate2.py")
    if not os.path.isfile(path):
        try:
            import rospkg

            path = os.path.join(rospkg.RosPack().get_path("mattbot_navigation"), "scripts", "localize_and_navigate2.py")
        except Exception:
            pass
    if not os.path.isfile(path):
        raise ImportError("Could not find localize_and_navigate2.py (tried next to %r)" % (__file__,))
    spec = importlib.util.spec_from_file_location("_localize_and_navigate2_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


l2 = _load_localize_and_navigate2()


class MultiagentComputingMode(IntEnum):
    """Single member; must not overlap localize_and_navigate2.Mode values (0–10)."""

    MULTIAGENT_CONTROL_COMPUTING = 11


class MultiAgentNavigator(l2.Navigator):
    def __init__(self, node_name="mattbot_navigator_multi_agent"):
        self._from_multi_robot_goal = False
        self._multi_plan_id = ""
        self._multi_coordinated = False
        self._multi_source_agent = 0
        self._peer_multi_planned_paths = {}
        self._multi_fleet_robot_ids = []
        self._multi_phase_started_at = None
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._simultaneous_lock = threading.Lock()
        self._timed_traj_armed = False
        self._pending_traj_times = None
        self._pending_traj = None
        self._execute_at_ros_time = None
        self._timed_arm_rostime = None
        self._auto_execute_timer = None
        super(MultiAgentNavigator, self).__init__(node_name=node_name)
        self._multi_path_wait_timeout = float(rospy.get_param("~multi_agent_path_wait_timeout", 60.0))
        self._multi_max_path_points = int(rospy.get_param("~multi_agent_max_path_points", 0))
        self._multi_agent_robot_diameter = float(rospy.get_param("~multi_agent_robot_diameter", 1.0))
        self._multi_agent_max_velocity = float(rospy.get_param("~multi_agent_max_velocity", 0.7))
        self._multi_agent_execute_max_lateness = float(rospy.get_param("~multi_agent_execute_max_lateness_sec", 5.0))
        self._multi_agent_execute_late_policy = rospy.get_param("~multi_agent_execute_late_policy", "immediate").strip().lower()
        self._multi_agent_execute_msg_wait_timeout = float(
            rospy.get_param("~multi_agent_execute_message_wait_timeout_sec", 120.0)
        )
        self._multi_agent_auto_execute = bool(rospy.get_param("~multi_agent_auto_execute", True))
        self._multi_agent_auto_execute_delay_sec = float(rospy.get_param("~multi_agent_auto_execute_delay_sec", 5.0))
        self._multi_agent_auto_execute_wall_extra_sec = float(
            rospy.get_param("~multi_agent_auto_execute_wall_extra_sec", 1.0)
        )
        self._leader_execute_dds_topic = rospy.get_param(
            "~multi_agent_execute_at_dds_trigger_topic", "/multi_agent_execute_at_dds"
        ).strip() or "/multi_agent_execute_at_dds"
        self._leader_execute_pub = rospy.Publisher(self._leader_execute_dds_topic, MultiAgentExecuteAt, queue_size=2, latch=False)
        topic = rospy.get_param("~external_goal_multi_topic", "/external_goal_multi").strip() or "/external_goal_multi"
        rospy.Subscriber(topic, MultiRobotExternalGoal, self.external_goal_multi_callback, queue_size=10)
        rospy.loginfo("MultiAgentNavigator: subscribed to %s", topic)

        self._dds_planned_pub_topic = rospy.get_param(
            "~multi_agent_planned_path_for_dds_topic", "/multi_agent_planned_path_for_dds"
        ).strip() or "/multi_agent_planned_path_for_dds"
        self._multi_agent_planned_path_pub = rospy.Publisher(
            self._dds_planned_pub_topic, MultiAgentPlannedPath, queue_size=2, latch=False
        )
        self._peer_planned_sub_topic = rospy.get_param(
            "~multi_agent_planned_path_from_agent_topic", "/multi_agent_planned_path_from_agent"
        ).strip() or "/multi_agent_planned_path_from_agent"
        rospy.Subscriber(
            self._peer_planned_sub_topic,
            MultiAgentPlannedPath,
            self._peer_multi_agent_planned_path_callback,
            queue_size=10,
        )
        rospy.loginfo(
            "MultiAgentNavigator: DDS planned path ROS topics pub=%s sub=%s",
            self._dds_planned_pub_topic,
            self._peer_planned_sub_topic,
        )
        self._execute_at_sub_topic = rospy.get_param("~multi_agent_execute_at_topic", "/multi_agent_execute_at").strip() or "/multi_agent_execute_at"
        rospy.Subscriber(self._execute_at_sub_topic, MultiAgentExecuteAt, self._multi_agent_execute_at_callback, queue_size=10)
        rospy.loginfo("MultiAgentNavigator: subscribed to execute_at topic %s", self._execute_at_sub_topic)
        rospy.loginfo(
            "MultiAgentNavigator: auto_execute=%s leader_delay_s=%.2f (fleet leader = first fleet_robot_ids)",
            self._multi_agent_auto_execute,
            self._multi_agent_auto_execute_delay_sec,
        )

    def _cancel_auto_execute_timer(self):
        if self._auto_execute_timer is not None:
            self._auto_execute_timer.shutdown()
            self._auto_execute_timer = None

    def _reset_timed_execute_state(self):
        self._cancel_auto_execute_timer()
        self._timed_traj_armed = False
        self._pending_traj_times = None
        self._pending_traj = None
        self._execute_at_ros_time = None
        self._timed_arm_rostime = None

    def _multi_agent_execute_at_callback(self, msg):
        if not self._timed_traj_armed:
            return
        if (msg.plan_id or "").strip() != (self._multi_plan_id or "").strip():
            return
        self._execute_at_ros_time = msg.execute_at
        rospy.loginfo(
            "MultiAgentNavigator: execute_at received plan_id=%s execute_at=%s",
            msg.plan_id,
            self._execute_at_ros_time,
        )

    def _maybe_arm_timed_trajectory(self):
        if not self._is_multiagent_computing_mode(self.mode) or not self._simultaneous_solve_done or self._timed_traj_armed:
            return
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        my_id = int(self.my_id)
        if my_id not in fleet:
            rospy.logwarn("MultiAgentNavigator: my_id %s not in fleet %s; aborting multi phase", my_id, fleet)
            self._abort_multi_to_idle()
            return
        ego_k = fleet.index(my_id)
        opt = self._simultaneous_optimized_times
        if opt is None or ego_k >= len(opt):
            rospy.logwarn("MultiAgentNavigator: missing optimized_times; aborting multi phase")
            self._abort_multi_to_idle()
            return
        t_wp = np.asarray(opt[ego_k], dtype=float)
        plan = getattr(self, "unsmoothed_plan", None) or []
        if len(t_wp) != len(plan):
            rospy.logwarn(
                "MultiAgentNavigator: len(optimized_times)=%d != len(plan)=%d; aborting",
                len(t_wp),
                len(plan),
            )
            self._abort_multi_to_idle()
            return
        if len(plan) < 4:
            rospy.loginfo("MultiAgentNavigator: timed path too short; PARK to goal")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.heading_controller.load_goal(self.theta_g)
            self.switch_mode(l2.Mode.PARK)
            self._reset_timed_execute_state()
            self._multi_plan_id = ""
            self._multi_fleet_robot_ids = []
            self._peer_multi_planned_paths = {}
            self._simultaneous_solve_done = False
            self._simultaneous_optimized_times = None
            return
        try:
            t_new, traj_new = compute_trajectory_from_timed_waypoints(
                plan, t_wp, self.spline_deg, self.spline_alpha, self.traj_dt
            )
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: timed trajectory build failed: %s", exc)
            self._abort_multi_to_idle()
            return
        self._pending_traj_times = t_new
        self._pending_traj = traj_new
        self._timed_traj_armed = True
        self._timed_arm_rostime = rospy.Time.now()
        self._execute_at_ros_time = None
        rospy.loginfo(
            "MultiAgentNavigator: armed timed trajectory (plan_id=%s duration_s=%.3f); waiting for execute_at",
            self._multi_plan_id,
            float(t_new[-1]) if len(t_new) else 0.0,
        )
        self._schedule_leader_auto_execute()

    def _schedule_leader_auto_execute(self):
        """First robot in fleet_robot_ids publishes execute_at after a delay (or disable via ~multi_agent_auto_execute)."""
        if not self._multi_agent_auto_execute:
            return
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if not fleet or int(self.my_id) != int(fleet[0]):
            return
        self._cancel_auto_execute_timer()
        d = max(0.0, self._multi_agent_auto_execute_delay_sec)
        rospy.loginfo(
            "MultiAgentNavigator: fleet leader robot %s scheduling auto execute_at publish in %.2fs -> %s",
            fleet[0],
            d,
            self._leader_execute_dds_topic,
        )
        self._auto_execute_timer = rospy.Timer(rospy.Duration(d), self._leader_auto_execute_timer_cb, oneshot=True)

    def _leader_auto_execute_timer_cb(self, _evt=None):
        self._auto_execute_timer = None
        if not self._is_multiagent_computing_mode(self.mode):
            return
        if not self._timed_traj_armed:
            return
        if self._execute_at_ros_time is not None:
            return
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if not fleet or int(self.my_id) != int(fleet[0]):
            return
        pid = (self._multi_plan_id or "").strip()
        if not pid:
            return
        msg = MultiAgentExecuteAt()
        msg.plan_id = pid
        msg.fleet_robot_ids = fleet
        msg.execute_at = rospy.Time.now() + rospy.Duration(max(0.05, self._multi_agent_auto_execute_wall_extra_sec))
        self._leader_execute_pub.publish(msg)
        rospy.loginfo(
            "MultiAgentNavigator: fleet leader published MultiAgentExecuteAt plan_id=%s execute_at=%s (DDS trigger %s)",
            pid,
            msg.execute_at,
            self._leader_execute_dds_topic,
        )

    def _commit_timed_trajectory(self):
        """Leave MULTI: load pending trajectory with t=0 at execute_at wall time; ALIGN or TRACK."""
        if not self._timed_traj_armed:
            return
        if self._pending_traj is None or self._pending_traj_times is None:
            rospy.logwarn("MultiAgentNavigator: commit called with no pending trajectory")
            self._abort_multi_to_idle()
            return
        if self._execute_at_ros_time is None:
            return
        now = rospy.Time.now()
        late_s = (now - self._execute_at_ros_time).to_sec()
        if late_s > self._multi_agent_execute_max_lateness and self._multi_agent_execute_late_policy == "idle":
            rospy.logwarn(
                "MultiAgentNavigator: execute_at late by %.2fs (max %.2fs, policy=idle) -> IDLE",
                late_s,
                self._multi_agent_execute_max_lateness,
            )
            self._abort_multi_to_idle()
            return
        t_new = self._pending_traj_times
        traj_new = self._pending_traj
        planned_path = getattr(self, "unsmoothed_plan", None) or []

        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub, times=t_new)
        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)
        self.current_plan = traj_new
        self.current_plan_start_time = self._execute_at_ros_time
        self.current_plan_duration = float(t_new[-1])

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        self.waypoints = []
        marker_arr = MarkerArray()
        for i in range(20, len(traj_new), 20):
            self.waypoints.append([traj_new[i, 0], traj_new[i, 1], t_new[i] + 2.0])
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

        self._simultaneous_solve_done = False
        self._simultaneous_optimized_times = None
        self._reset_timed_execute_state()

        if not self.aligned():
            rospy.loginfo("MultiAgentNavigator: timed plan — not aligned with start direction -> ALIGN")
            self.switch_mode(l2.Mode.ALIGN)
        else:
            rospy.loginfo("MultiAgentNavigator: timed plan -> TRACK")
            self.switch_mode(l2.Mode.TRACK)

    def _build_sp_occupancy_grid(self):
        """`social_path_planning` grid copy for MultiAgentSimultaneousPlanner (same geometry as self.occupancy)."""
        occ = getattr(self, "occupancy", None)
        if occ is None:
            return None
        return SpStochOccupancyGrid2D(
            occ.resolution,
            occ.width,
            occ.height,
            occ.origin_x,
            occ.origin_y,
            occ.window_size,
            np.asarray(occ.probs),
            thresh=float(occ.thresh),
            robot_d=float(occ.robot_d),
            wall_distance_cache_path=None,
            auto_build_wall_distance_cache=False,
        )

    def _maybe_downsample_path_xy(self, pts):
        m = self._multi_max_path_points
        if m <= 0 or len(pts) <= m:
            return pts
        idx = np.unique(np.linspace(0, len(pts) - 1, num=m, dtype=int))
        return [pts[i] for i in idx]

    def _assemble_paths_for_simultaneous(self):
        paths = []
        for rid in self._multi_fleet_robot_ids:
            rid = int(rid)
            if rid == int(self.my_id):
                plan = getattr(self, "unsmoothed_plan", None) or []
                pts = [(float(s[0]), float(s[1])) for s in plan]
            else:
                peer = self._peer_multi_planned_paths[rid]
                pts = [
                    (float(ps.pose.position.x), float(ps.pose.position.y)) for ps in peer.path.poses
                ]
            paths.append(self._maybe_downsample_path_xy(pts))
        return paths

    def _fleet_paths_complete(self):
        pid = (self._multi_plan_id or "").strip()
        if not pid:
            return False
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if not fleet:
            return False
        ego_plan = getattr(self, "unsmoothed_plan", None) or []
        if len(ego_plan) == 0:
            return False
        my_id = int(self.my_id)
        for rid in fleet:
            if rid == my_id:
                continue
            p = self._peer_multi_planned_paths.get(rid)
            if p is None or (p.plan_id or "").strip() != pid or len(p.path.poses) == 0:
                return False
        return True

    def _abort_multi_to_idle(self):
        rospy.logwarn("MultiAgentNavigator: aborting multi-agent phase -> IDLE")
        self._multi_plan_id = ""
        self._multi_coordinated = False
        self._multi_source_agent = 0
        self._multi_fleet_robot_ids = []
        self._peer_multi_planned_paths = {}
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._multi_phase_started_at = None
        self._reset_timed_execute_state()
        self.switch_mode(l2.Mode.IDLE)

    def _try_simultaneous_plan_if_ready(self):
        if not self._is_multiagent_computing_mode(self.mode):
            return
        with self._simultaneous_lock:
            if self._simultaneous_solve_done or self._simultaneous_solve_running or self._simultaneous_solve_failed:
                return
            if not self._fleet_paths_complete():
                return
            self._simultaneous_solve_running = True
        rospy.loginfo(
            "MultiAgentNavigator: starting simultaneous timing solve (plan_id=%s fleet=%s)",
            self._multi_plan_id,
            self._multi_fleet_robot_ids,
        )
        threading.Thread(target=self._simultaneous_planner_thread_main, daemon=True).start()

    def _simultaneous_planner_thread_main(self):
        try:
            import social_path_planning.multi_planning as smp

            smp.ROBOT_DIAMETER = float(self._multi_agent_robot_diameter)
            smp.MAX_VELOCITY = float(self._multi_agent_max_velocity)
            occ = self._build_sp_occupancy_grid()
            if occ is None:
                raise RuntimeError("occupancy grid missing for simultaneous planner")
            paths = self._assemble_paths_for_simultaneous()
            if not paths or any(len(p) < 1 for p in paths):
                raise RuntimeError("invalid paths for simultaneous planner")
            v_list = [float(self._multi_agent_max_velocity)] * len(paths)
            planner = MultiAgentSimultaneousPlanner(occ, paths=paths, norm=1, v=v_list)
            optimized_times = planner.plan()
            lens = [len(t) for t in optimized_times] if optimized_times else []
            rospy.loginfo(
                "MultiAgentNavigator: simultaneous plan solved plan_id=%s waypoint_time_lens=%s",
                self._multi_plan_id,
                lens,
            )
            self._simultaneous_optimized_times = optimized_times
            self._simultaneous_solve_done = True
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: simultaneous plan failed: %s", exc)
            self._simultaneous_solve_failed = True
        finally:
            self._simultaneous_solve_running = False

    @staticmethod
    def _is_multiagent_computing_mode(mode):
        return isinstance(mode, MultiagentComputingMode)

    def _path_from_unsmoothed_plan(self):
        """nav_msgs/Path in map frame from self.unsmoothed_plan (same layout as publish_planned_path)."""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        plan = getattr(self, "unsmoothed_plan", None) or []
        for state in plan:
            pose_st = PoseStamped()
            pose_st.header.frame_id = "map"
            pose_st.header.stamp = path_msg.header.stamp
            pose_st.pose.position.x = float(state[0])
            pose_st.pose.position.y = float(state[1])
            pose_st.pose.orientation.w = 1.0
            path_msg.poses.append(pose_st)
        return path_msg

    def _publish_planned_path_for_multi_dds(self):
        """ROS topic consumed by dds_data_publisher -> DDS multi_agent_planned_path."""
        path_msg = self._path_from_unsmoothed_plan()
        if not path_msg.poses:
            rospy.logwarn("MultiAgentNavigator: unsmoothed plan empty; skipping DDS planned path publish")
            return
        out = MultiAgentPlannedPath()
        out.plan_id = self._multi_plan_id
        out.source_agent = int(self.my_id)
        out.path = path_msg
        self._multi_agent_planned_path_pub.publish(out)
        rospy.loginfo(
            "MultiAgentNavigator: published MultiAgentPlannedPath for DDS (%d poses) plan_id=%s",
            len(path_msg.poses),
            self._multi_plan_id,
        )
        self._try_simultaneous_plan_if_ready()

    def _peer_multi_agent_planned_path_callback(self, msg):
        if int(msg.source_agent) == int(self.my_id):
            return
        pid = (self._multi_plan_id or "").strip()
        if not pid or (msg.plan_id or "").strip() != pid:
            return
        self._peer_multi_planned_paths[int(msg.source_agent)] = msg
        n_poses = len(msg.path.poses)
        rospy.loginfo(
            "MultiAgentNavigator: received peer planned path source_agent=%d plan_id=%s poses=%d",
            int(msg.source_agent),
            msg.plan_id,
            n_poses
        )
        self._try_simultaneous_plan_if_ready()

    def external_goal_multi_callback(self, msg):
        """Same policy as external_goal_callback, but pose from MultiRobotExternalGoal + plan metadata."""
        g = msg.goal
        if (
            self.x_g is not None
            and self.y_g is not None
            and self.theta_g is not None
            and g.x == self.x_g
            and g.y == self.y_g
            and g.theta == self.theta_g
            and msg.plan_id == getattr(self, "_multi_plan_id", "")
        ):
            rospy.loginfo("Multi external goal unchanged (pose + plan_id), ignoring")
            return

        if self.mode == l2.Mode.WAITING_FOR_INIT or self.mode == l2.Mode.LOCALIZING:
            return

        if self.mode != l2.Mode.IDLE:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            self.switch_mode(l2.Mode.IDLE)

        if self.occupancy is not None and not self.occupancy.is_free((g.x, g.y)):
            rospy.loginfo("Multi external goal: not a valid goal cell")
            invalid_goal_msg = Pose2D()
            invalid_goal_msg.x = g.x
            invalid_goal_msg.y = g.y
            invalid_goal_msg.theta = g.theta
            self.invalid_goal_pub.publish(invalid_goal_msg)
            return

        self._peer_multi_planned_paths = {}
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._multi_phase_started_at = None
        self._reset_timed_execute_state()

        self._from_multi_robot_goal = True
        self._multi_plan_id = msg.plan_id
        self._multi_coordinated = bool(msg.coordinated)
        self._multi_source_agent = int(msg.source_agent)
        self._multi_fleet_robot_ids = [int(x) for x in (msg.fleet_robot_ids or [])]

        self.x_g = g.x
        self.y_g = g.y
        self.theta_g = g.theta
        rospy.loginfo(
            "Multi external goal: plan_id=%s coordinated=%s source=%s -> replan",
            self._multi_plan_id,
            self._multi_coordinated,
            self._multi_source_agent,
        )
        self.replan()

    def replan(self, obj_x=None, obj_y=None, obj_d=None):
        if obj_x is None:
            obj_x = []
        if obj_y is None:
            obj_y = []
        if obj_d is None:
            obj_d = []

        if self._is_multiagent_computing_mode(self.mode):
            rospy.logdebug_throttle(2.0, "replan ignored in MULTIAGENT_CONTROL_COMPUTING")
            return

        multi = getattr(self, "_from_multi_robot_goal", False)
        super(MultiAgentNavigator, self).replan(obj_x, obj_y, obj_d)

        if multi:
            self._from_multi_robot_goal = False
            if self.mode in (l2.Mode.ALIGN, l2.Mode.TRACK, l2.Mode.PARK):
                self.switch_mode(MultiagentComputingMode.MULTIAGENT_CONTROL_COMPUTING)
                rospy.loginfo(
                    "Entering MULTIAGENT_CONTROL_COMPUTING after plan (plan_id=%s)",
                    self._multi_plan_id,
                )
                self._multi_phase_started_at = rospy.Time.now()
                self._publish_planned_path_for_multi_dds()
            elif self.mode == l2.Mode.IDLE:
                self._multi_plan_id = ""
                self._multi_coordinated = False
                self._multi_source_agent = 0
                self._multi_fleet_robot_ids = []
                self._reset_timed_execute_state()

    def publish_control(self):
        if self._is_multiagent_computing_mode(self.mode):
            if self._simultaneous_solve_failed:
                self._abort_multi_to_idle()
                return
            if (
                not self._simultaneous_solve_done
                and not self._simultaneous_solve_running
                and not self._fleet_paths_complete()
                and self._multi_phase_started_at is not None
            ):
                elapsed = (rospy.Time.now() - self._multi_phase_started_at).to_sec()
                if elapsed > self._multi_path_wait_timeout:
                    rospy.logwarn(
                        "MultiAgentNavigator: path wait timeout (%.1fs) plan_id=%s fleet=%s -> IDLE",
                        self._multi_path_wait_timeout,
                        getattr(self, "_multi_plan_id", "") or "?",
                        self._multi_fleet_robot_ids,
                    )
                    self._abort_multi_to_idle()
                    return
            self._try_simultaneous_plan_if_ready()
            if self._simultaneous_solve_done:
                self._maybe_arm_timed_trajectory()

            if self._timed_traj_armed and self._execute_at_ros_time is None and self._timed_arm_rostime is not None:
                wait_s = (rospy.Time.now() - self._timed_arm_rostime).to_sec()
                if wait_s > self._multi_agent_execute_msg_wait_timeout:
                    rospy.logwarn(
                        "MultiAgentNavigator: no execute_at message after %.1fs -> IDLE",
                        self._multi_agent_execute_msg_wait_timeout,
                    )
                    self._abort_multi_to_idle()
                    return

            if (
                self._timed_traj_armed
                and self._execute_at_ros_time is not None
                and rospy.Time.now() >= self._execute_at_ros_time
            ):
                self._commit_timed_trajectory()
                super(MultiAgentNavigator, self).publish_control()
                return

            msg = "holding cmd_vel=0"
            if self._simultaneous_solve_done and not self._timed_traj_armed:
                msg = "simultaneous timing solved; building trajectory"
            elif self._timed_traj_armed and self._execute_at_ros_time is None:
                msg = "armed; waiting for execute_at message (DDS)"
            elif self._timed_traj_armed:
                msg = "armed; waiting for execute_at wall time"
            rospy.loginfo_throttle(
                5.0,
                "MULTIAGENT_CONTROL_COMPUTING (plan_id=%s); %s",
                getattr(self, "_multi_plan_id", "") or "?",
                msg,
            )
            self.prev_om = 0.0
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            return
        super(MultiAgentNavigator, self).publish_control()


if __name__ == "__main__":
    nav = MultiAgentNavigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(3)
    nav.run()
