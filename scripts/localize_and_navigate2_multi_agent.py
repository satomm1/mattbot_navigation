#!/usr/bin/env python3
"""
Navigator variant: same behavior as localize_and_navigate2, plus /external_goal_multi
(MultiRobotExternalGoal) and singleton coordination. After a successful plan from a multi-robot goal,
runs a pre-MULTI ALIGN dwell, then MULTIAGENT_CONTROL_COMPUTING: publishes planned path, then either
(1) fleet size >= 2: coordinator runs simultaneous MILP and MultiAgentTimingSolve over DDS;
(2) fleet size == 1 with peer active trajectories in cache: coordinator runs MultiAgentSequentialPlanner
with shifted global times and fixed execute_at = now + budget;
(3) fleet size == 1 solo: analytic max-velocity waypoint times (no MILP).

For missions started via /external_goal_multi, after the first timed TRACK commit, further replans
(e.g. plan duration elapsed or waypoint deadlines while moving) re-enter the same timing pipeline.
If the mission was coordinated with fleet size >= 2, those sticky replans shrink to ego-only timing
(sequential when peer active trajectories exist in cache, else solo), not a second simultaneous MILP.

Peers publish MultiAgentActiveTrajectory snapshots on timed TRACK commit; cache is pruned periodically
and on inactive messages. /external_goal (Pose2D) and /voice_goal use the same singleton path as a
one-robot multi goal (plan_id solo_pose).

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
from mattbot_dds.msg import (
    MultiAgentActiveTrajectory,
    MultiAgentExecuteAt,
    MultiAgentPlannedPath,
    MultiAgentTimingSolve,
    MultiRobotExternalGoal,
)
from navigation_utils import compute_trajectory_from_timed_waypoints
from social_path_planning.occupancy_grid import StochOccupancyGrid2D as SpStochOccupancyGrid2D
from visualization_msgs.msg import Marker, MarkerArray
from social_path_planning.multi_planning import MultiAgentSequentialPlanner, MultiAgentSimultaneousPlanner


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
        # True only for goals from /external_goal_multi; used to arm sticky replans after first TRACK commit.
        self._replan_as_multi_from_external_goal_multi = False
        # After a timed TRACK commit from /external_goal_multi: sticky replans re-enter MULTI timing (see module doc).
        self._sticky_multi_timing_replan = False
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
        self._awaiting_pre_multi_align = False
        self._pre_multi_align_started_at = None
        self._timing_solve_wait_started_at = None
        self._peer_active_trajectories = {}
        self._peer_traj_cache_lock = threading.Lock()
        self._pending_execute_at_for_arm = None
        self._armed_waypoint_times_for_snapshot = None
        self._active_traj_prune_timer = None
        super(MultiAgentNavigator, self).__init__(node_name=node_name)
        self._multi_path_wait_timeout = float(rospy.get_param("~multi_agent_path_wait_timeout", 60.0))
        self._multi_agent_timing_solve_wait_timeout_sec = float(
            rospy.get_param("~multi_agent_timing_solve_wait_timeout_sec", 120.0)
        )
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
        self._timing_solve_for_dds_topic = rospy.get_param(
            "~multi_agent_timing_solve_for_dds_topic", "/multi_agent_timing_solve_for_dds"
        ).strip() or "/multi_agent_timing_solve_for_dds"
        self._multi_agent_timing_solve_for_dds_pub = rospy.Publisher(
            self._timing_solve_for_dds_topic, MultiAgentTimingSolve, queue_size=2, latch=False
        )
        self._timing_solve_sub_topic = rospy.get_param(
            "~multi_agent_timing_solve_topic", "/multi_agent_timing_solve"
        ).strip() or "/multi_agent_timing_solve"
        rospy.Subscriber(
            self._timing_solve_sub_topic,
            MultiAgentTimingSolve,
            self._multi_agent_timing_solve_callback,
            queue_size=10,
        )
        rospy.loginfo(
            "MultiAgentNavigator: timing solve DDS pub=%s sub=%s",
            self._timing_solve_for_dds_topic,
            self._timing_solve_sub_topic,
        )
        rospy.loginfo(
            "MultiAgentNavigator: auto_execute=%s coordinator_delay_s=%.2f (coordinator = min fleet_robot_ids)",
            self._multi_agent_auto_execute,
            self._multi_agent_auto_execute_delay_sec,
        )
        self._pre_multi_align_sec = float(rospy.get_param("~multi_agent_pre_multi_align_sec", 2.0))
        self._pre_multi_align_exit_policy = rospy.get_param(
            "~multi_agent_pre_multi_align_exit_policy", "aligned_or_max"
        ).strip().lower()
        if self._pre_multi_align_exit_policy not in ("aligned_or_max", "max_only"):
            rospy.logwarn(
                "MultiAgentNavigator: unknown ~multi_agent_pre_multi_align_exit_policy=%r; using aligned_or_max",
                self._pre_multi_align_exit_policy,
            )
            self._pre_multi_align_exit_policy = "aligned_or_max"
        rospy.loginfo(
            "MultiAgentNavigator: pre_multi_align max_s=%.2f exit_policy=%s",
            self._pre_multi_align_sec,
            self._pre_multi_align_exit_policy,
        )
        rospy.loginfo(
            "MultiAgentNavigator: coordinated MILP runs on min(fleet_robot_ids); timing via MultiAgentTimingSolve DDS"
        )
        self._multi_agent_sequential_budget_sec = float(rospy.get_param("~multi_agent_sequential_budget_sec", 3.0))
        self._multi_agent_active_traj_ttl_sec = float(rospy.get_param("~multi_agent_active_trajectory_ttl_sec", 120.0))
        self._multi_agent_active_traj_check_period_sec = float(
            rospy.get_param("~multi_agent_active_trajectory_check_period_sec", 1.0)
        )
        self._multi_agent_trajectory_done_grace_sec = float(
            rospy.get_param("~multi_agent_trajectory_done_grace_sec", 0.5)
        )
        self._multi_agent_active_traj_forward_extra_ids = [
            int(x) for x in rospy.get_param("~multi_agent_active_trajectory_forward_ids", [])
        ]
        self._active_traj_for_dds_topic = rospy.get_param(
            "~multi_agent_active_trajectory_for_dds_topic", "/multi_agent_active_trajectory_for_dds"
        ).strip() or "/multi_agent_active_trajectory_for_dds"
        self._active_traj_pub = rospy.Publisher(
            self._active_traj_for_dds_topic, MultiAgentActiveTrajectory, queue_size=2, latch=False
        )
        self._active_traj_sub_topic = rospy.get_param(
            "~multi_agent_active_trajectory_topic", "/multi_agent_active_trajectory"
        ).strip() or "/multi_agent_active_trajectory"
        rospy.Subscriber(
            self._active_traj_sub_topic,
            MultiAgentActiveTrajectory,
            self._peer_active_trajectory_callback,
            queue_size=10,
        )
        rospy.loginfo(
            "MultiAgentNavigator: active trajectory pub=%s sub=%s sequential_budget_s=%.2f",
            self._active_traj_for_dds_topic,
            self._active_traj_sub_topic,
            self._multi_agent_sequential_budget_sec,
        )
        self._active_traj_prune_timer = rospy.Timer(
            rospy.Duration(max(0.2, self._multi_agent_active_traj_check_period_sec)),
            self._peer_active_trajectory_prune_timer_cb,
        )

    def external_goal_callback(self, msg):
        """Pose2D / voice goal: treat as singleton fleet coordination (sequential vs solo like one-robot multi goal)."""
        if (
            self.x_g is not None
            and self.y_g is not None
            and self.theta_g is not None
            and (msg.x == self.x_g and msg.y == self.y_g and msg.theta == self.theta_g)
        ):
            rospy.loginfo("External goal is the same as current goal, ignoring")
            return

        if self.mode == l2.Mode.WAITING_FOR_INIT or self.mode == l2.Mode.LOCALIZING:
            return

        if self.mode != l2.Mode.IDLE:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            self.switch_mode(l2.Mode.IDLE)

        if self.occupancy is not None and not self.occupancy.is_free((msg.x, msg.y)):
            rospy.loginfo("Not a valid goal")
            invalid_goal_msg = Pose2D()
            invalid_goal_msg.x = msg.x
            invalid_goal_msg.y = msg.y
            invalid_goal_msg.theta = msg.theta
            self.invalid_goal_pub.publish(invalid_goal_msg)
            return

        self._peer_multi_planned_paths = {}
        self._reset_timing_compute_state()

        self._replan_as_multi_from_external_goal_multi = False
        self._sticky_multi_timing_replan = False
        self._from_multi_robot_goal = True
        self._multi_plan_id = "solo_pose"
        self._multi_coordinated = False
        self._multi_source_agent = int(self.my_id)
        self._multi_fleet_robot_ids = [int(self.my_id)]

        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        rospy.loginfo("MultiAgentNavigator: external_goal -> singleton fleet replan plan_id=%s", self._multi_plan_id)
        self.replan()

    def _clear_pre_multi_align_state(self):
        self._awaiting_pre_multi_align = False
        self._pre_multi_align_started_at = None

    def _reset_timing_compute_state(self):
        """Clear simultaneous/sequential thread state between goals or replan cycles."""
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._multi_phase_started_at = None
        self._reset_timed_execute_state()
        self._clear_pre_multi_align_state()
        self._timing_solve_wait_started_at = None

    def _fleet_coordinator_id(self):
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        return min(fleet) if fleet else None

    def _i_am_coordinator(self):
        cid = self._fleet_coordinator_id()
        return cid is not None and int(self.my_id) == cid

    @staticmethod
    def _unpack_timing_solve_flat(counts, flat):
        rows = []
        idx = 0
        for c in counts:
            c = int(c)
            if c < 0 or idx + c > len(flat):
                return None
            rows.append([float(x) for x in flat[idx : idx + c]])
            idx += c
        if idx != len(flat):
            return None
        return rows

    def _path_msg_to_xy(self, path_msg):
        out = []
        for ps in path_msg.poses:
            out.append((float(ps.pose.position.x), float(ps.pose.position.y)))
        return out

    def _solo_velocity_waypoint_times(self, plan):
        """Cumulative relative times along polyline at max velocity (no MILP)."""
        if not plan or len(plan) < 2:
            return [0.0] * max(1, len(plan))
        v = float(self._multi_agent_max_velocity)
        if v <= 0:
            v = 0.5
        t = [0.0]
        acc = 0.0
        for i in range(len(plan) - 1):
            dx = float(plan[i + 1][0]) - float(plan[i][0])
            dy = float(plan[i + 1][1]) - float(plan[i][1])
            acc += float(np.hypot(dx, dy)) / v
            t.append(acc)
        return t

    def _peer_active_obstacles_available(self):
        with self._peer_traj_cache_lock:
            for rid, rec in list(self._peer_active_trajectories.items()):
                if int(rid) == int(self.my_id):
                    continue
                if not rec.get("active", True):
                    continue
                if len(rec.get("path_xy") or []) < 1:
                    continue
                ta = rec.get("waypoint_times") or []
                if len(ta) != len(rec.get("path_xy") or []):
                    continue
                return True
        return False

    def _peer_active_trajectory_callback(self, msg):
        if int(msg.robot_id) == int(self.my_id):
            return
        if not msg.active and int(msg.robot_id) in self._peer_active_trajectories:
            with self._peer_traj_cache_lock:
                self._peer_active_trajectories.pop(int(msg.robot_id), None)
            rospy.loginfo("MultiAgentNavigator: peer %s active trajectory cleared (inactive)", msg.robot_id)
            return
        path_xy = self._path_msg_to_xy(msg.path)
        wt = [float(x) for x in (msg.waypoint_times or [])]
        if len(path_xy) == 0 or len(wt) != len(path_xy):
            rospy.logwarn_throttle(5.0, "MultiAgentNavigator: ignoring active_trajectory robot=%s len mismatch", msg.robot_id)
            return
        with self._peer_traj_cache_lock:
            self._peer_active_trajectories[int(msg.robot_id)] = {
                "path_xy": path_xy,
                "waypoint_times": wt,
                "execute_at": msg.execute_at,
                "stamp": rospy.Time.now(),
                "plan_id": str(msg.plan_id or ""),
                "active": bool(msg.active),
            }
        rospy.loginfo(
            "MultiAgentNavigator: cached active trajectory robot=%s wp=%d plan_id=%s",
            msg.robot_id,
            len(path_xy),
            msg.plan_id,
        )

    def _peer_active_trajectory_prune_timer_cb(self, _evt=None):
        now = rospy.Time.now()
        remove = []
        with self._peer_traj_cache_lock:
            for rid, rec in list(self._peer_active_trajectories.items()):
                if int(rid) == int(self.my_id):
                    remove.append(rid)
                    continue
                if not rec.get("active", True):
                    remove.append(rid)
                    continue
                st = rec.get("stamp")
                if st is not None and (now - st).to_sec() > self._multi_agent_active_traj_ttl_sec:
                    remove.append(rid)
                    continue
                ex = rec.get("execute_at")
                ta = rec.get("waypoint_times") or []
                if ex is not None and ta:
                    t_end = float(ta[-1])
                    if now.to_sec() > ex.to_sec() + t_end + self._multi_agent_trajectory_done_grace_sec:
                        remove.append(rid)
        if remove:
            with self._peer_traj_cache_lock:
                for rid in remove:
                    self._peer_active_trajectories.pop(rid, None)

    def _dds_forward_ids_for_active_traj(self):
        ids = set(int(x) for x in self._multi_agent_active_traj_forward_extra_ids)
        with self._peer_traj_cache_lock:
            ids.update(int(k) for k in self._peer_active_trajectories.keys())
        for rid in self._multi_fleet_robot_ids:
            ids.add(int(rid))
        ids.discard(int(self.my_id))
        return sorted(ids)

    def _publish_active_trajectory_msg(self, active, path_msg, waypoint_times, execute_at, plan_id):
        out = MultiAgentActiveTrajectory()
        out.robot_id = int(self.my_id)
        out.plan_id = str(plan_id or "")
        out.active = bool(active)
        out.execute_at = execute_at if execute_at is not None else rospy.Time(0)
        out.path = path_msg
        out.waypoint_times = [float(x) for x in waypoint_times]
        out.dds_forward_robot_ids = self._dds_forward_ids_for_active_traj()
        self._active_traj_pub.publish(out)

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

    def _multi_agent_timing_solve_callback(self, msg):
        """Apply MILP timing from coordinator (DDS -> ROS); coordinator skips local echo (already set in solver thread)."""
        if not self._is_multiagent_computing_mode(self.mode):
            return
        pid = (msg.plan_id or "").strip()
        if pid != (self._multi_plan_id or "").strip():
            return
        coord = self._fleet_coordinator_id()
        if coord is None or int(msg.source_agent) != coord:
            rospy.logwarn_throttle(
                5.0,
                "MultiAgentNavigator: ignoring timing_solve source=%s expected coordinator=%s",
                msg.source_agent,
                coord,
            )
            return
        if self._i_am_coordinator() and int(msg.source_agent) == int(self.my_id):
            return

        fleet_msg = [int(x) for x in (msg.fleet_robot_ids or [])]
        fleet_local = [int(x) for x in self._multi_fleet_robot_ids]
        if fleet_msg != fleet_local:
            rospy.logwarn(
                "MultiAgentNavigator: timing_solve fleet_robot_ids %s != local %s; aborting multi",
                fleet_msg,
                fleet_local,
            )
            self._simultaneous_solve_failed = True
            return

        counts = [int(x) for x in (msg.waypoint_counts or [])]
        flat = [float(x) for x in (msg.waypoint_times_flat or [])]
        rows = self._unpack_timing_solve_flat(counts, flat)
        if rows is None or len(rows) != len(fleet_local):
            rospy.logwarn("MultiAgentNavigator: invalid timing_solve payload; aborting multi")
            self._simultaneous_solve_failed = True
            return

        my_id = int(self.my_id)
        ego_k = fleet_local.index(my_id)
        plan = getattr(self, "unsmoothed_plan", None) or []
        if ego_k >= len(rows) or len(rows[ego_k]) != len(plan):
            rospy.logwarn(
                "MultiAgentNavigator: ego timing len=%s vs plan len=%d; aborting multi",
                len(rows[ego_k]) if ego_k < len(rows) else None,
                len(plan),
            )
            self._simultaneous_solve_failed = True
            return

        with self._simultaneous_lock:
            if self._simultaneous_solve_done:
                return
            self._simultaneous_optimized_times = rows
            self._simultaneous_solve_done = True
            self._simultaneous_solve_running = False
        self._timing_solve_wait_started_at = None
        rospy.loginfo(
            "MultiAgentNavigator: applied timing_solve from coordinator=%s plan_id=%s ego_wp=%d",
            msg.source_agent,
            pid,
            len(rows[ego_k]),
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
            self._reset_timing_compute_state()
            self._multi_plan_id = ""
            self._multi_fleet_robot_ids = []
            self._peer_multi_planned_paths = {}
            self._sticky_multi_timing_replan = False
            self._replan_as_multi_from_external_goal_multi = False
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
        if self._pending_execute_at_for_arm is not None:
            self._execute_at_ros_time = self._pending_execute_at_for_arm
            self._pending_execute_at_for_arm = None
        else:
            self._execute_at_ros_time = None
            self._schedule_leader_auto_execute()
        rospy.loginfo(
            "MultiAgentNavigator: armed timed trajectory (plan_id=%s duration_s=%.3f); waiting for execute_at",
            self._multi_plan_id,
            float(t_new[-1]) if len(t_new) else 0.0,
        )

    def _schedule_leader_auto_execute(self):
        """Coordinator (min fleet_robot_ids) publishes execute_at after a delay (or disable via ~multi_agent_auto_execute)."""
        if not self._multi_agent_auto_execute:
            return
        if self._execute_at_ros_time is not None:
            return
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        coord = self._fleet_coordinator_id()
        if coord is None or int(self.my_id) != coord:
            return
        self._cancel_auto_execute_timer()
        # Allow peers to finish pre-MULTI ALIGN + one MILP + DDS; increase if fleet is large or clocks loose.
        d = max(0.0, self._multi_agent_auto_execute_delay_sec)
        rospy.loginfo(
            "MultiAgentNavigator: coordinator robot %s scheduling auto execute_at publish in %.2fs -> %s",
            coord,
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
        coord = self._fleet_coordinator_id()
        if coord is None or int(self.my_id) != coord:
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
            "MultiAgentNavigator: coordinator published MultiAgentExecuteAt plan_id=%s execute_at=%s (DDS trigger %s)",
            pid,
            msg.execute_at,
            self._leader_execute_dds_topic,
        )

    def _commit_timed_trajectory(self):
        """Leave MULTI: load pending trajectory with t=0 at execute_at wall time; TRACK (pre-MULTI ALIGN already done)."""
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

        wt_pub = self._armed_waypoint_times_for_snapshot
        if wt_pub is None or len(wt_pub) != len(planned_path):
            wt_pub = self._solo_velocity_waypoint_times(planned_path)
        path_snap = Path()
        path_snap.header.frame_id = "map"
        path_snap.header.stamp = rospy.Time.now()
        for state in planned_path:
            ps = PoseStamped()
            ps.header = path_snap.header
            ps.pose.position.x = float(state[0])
            ps.pose.position.y = float(state[1])
            ps.pose.orientation.w = 1.0
            path_snap.poses.append(ps)
        self._publish_active_trajectory_msg(True, path_snap, wt_pub, self._execute_at_ros_time, self._multi_plan_id)

        self._simultaneous_solve_done = False
        self._simultaneous_optimized_times = None
        self._armed_waypoint_times_for_snapshot = None
        self._reset_timed_execute_state()
        self._clear_pre_multi_align_state()
        self._timing_solve_wait_started_at = None

        if not self.aligned():
            rospy.logwarn(
                "MultiAgentNavigator: timed plan commit while not aligned with start heading; TRACK anyway (pre-MULTI ALIGN should have handled this)"
            )
        rospy.loginfo("MultiAgentNavigator: timed plan -> TRACK")
        self.switch_mode(l2.Mode.TRACK)
        if self._replan_as_multi_from_external_goal_multi:
            self._sticky_multi_timing_replan = True
            rospy.loginfo(
                "MultiAgentNavigator: sticky multi timing replan enabled (plan_id=%s) for timeout / waypoint-miss replans",
                self._multi_plan_id or "?",
            )

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

    def _downsample_path_xy_and_times(self, pts, times):
        m = self._multi_max_path_points
        if m <= 0 or len(pts) <= m or len(pts) != len(times):
            return pts, times
        idx = np.unique(np.linspace(0, len(pts) - 1, num=m, dtype=int))
        return [pts[i] for i in idx], [times[i] for i in idx]

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
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if not fleet:
            return False
        ego_plan = getattr(self, "unsmoothed_plan", None) or []
        if len(ego_plan) == 0:
            return False
        if not pid:
            return False
        if len(fleet) == 1:
            return True
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
        try:
            empty = Path()
            empty.header.frame_id = "map"
            empty.header.stamp = rospy.Time.now()
            self._publish_active_trajectory_msg(False, empty, [], rospy.Time(0), self._multi_plan_id)
        except Exception:
            pass
        self._multi_plan_id = ""
        self._multi_coordinated = False
        self._multi_source_agent = 0
        self._multi_fleet_robot_ids = []
        self._sticky_multi_timing_replan = False
        self._replan_as_multi_from_external_goal_multi = False
        self._peer_multi_planned_paths = {}
        self._reset_timing_compute_state()
        self.switch_mode(l2.Mode.IDLE)

    def _try_simultaneous_plan_if_ready(self):
        if not self._is_multiagent_computing_mode(self.mode):
            return
        if not self._i_am_coordinator():
            return
        with self._simultaneous_lock:
            if self._simultaneous_solve_done or self._simultaneous_solve_running or self._simultaneous_solve_failed:
                return
            if not self._fleet_paths_complete():
                return
            self._simultaneous_solve_running = True
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if len(fleet) >= 2:
            rospy.loginfo(
                "MultiAgentNavigator: coordinator starting simultaneous timing solve (plan_id=%s fleet=%s)",
                self._multi_plan_id,
                self._multi_fleet_robot_ids,
            )
            threading.Thread(target=self._simultaneous_planner_thread_main, daemon=True).start()
        elif len(fleet) == 1 and self._peer_active_obstacles_available():
            rospy.loginfo(
                "MultiAgentNavigator: starting sequential timing solve (plan_id=%s peers in cache)",
                self._multi_plan_id,
            )
            threading.Thread(target=self._sequential_planner_thread_main, daemon=True).start()
        else:
            rospy.loginfo(
                "MultiAgentNavigator: singleton fleet solo timing (plan_id=%s no peer trajectories)",
                self._multi_plan_id,
            )
            threading.Thread(target=self._solo_timing_thread_main, daemon=True).start()

    def _solo_timing_thread_main(self):
        try:
            plan = getattr(self, "unsmoothed_plan", None) or []
            if len(plan) < 1:
                raise RuntimeError("empty plan")
            times = self._solo_velocity_waypoint_times(plan)
            self._simultaneous_optimized_times = [times]
            self._armed_waypoint_times_for_snapshot = list(times)
            T_ego = rospy.Time.now() + rospy.Duration(max(0.05, self._multi_agent_sequential_budget_sec))
            self._pending_execute_at_for_arm = T_ego
            rospy.loginfo(
                "MultiAgentNavigator: solo timing ready plan_id=%s T_ego=%s duration_s=%.3f",
                self._multi_plan_id,
                T_ego,
                float(times[-1]) if times else 0.0,
            )
            self._simultaneous_solve_done = True
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: solo timing failed: %s", exc)
            self._simultaneous_solve_failed = True
        finally:
            self._simultaneous_solve_running = False

    def _sequential_planner_thread_main(self):
        try:
            import social_path_planning.multi_planning as smp

            smp.ROBOT_DIAMETER = float(self._multi_agent_robot_diameter)
            smp.MAX_VELOCITY = float(self._multi_agent_max_velocity)
            T_ego = rospy.Time.now() + rospy.Duration(max(0.05, self._multi_agent_sequential_budget_sec))
            t_ego_sec = T_ego.to_sec()
            ego_plan = getattr(self, "unsmoothed_plan", None) or []
            ego_path = [(float(s[0]), float(s[1])) for s in ego_plan]
            other_paths = []
            other_times_shifted = []
            with self._peer_traj_cache_lock:
                snap = {int(k): dict(v) for k, v in self._peer_active_trajectories.items()}
            for rid, rec in sorted(snap.items()):
                if rid == int(self.my_id):
                    continue
                if not rec.get("active", True):
                    continue
                pxy = rec.get("path_xy") or []
                tau = rec.get("waypoint_times") or []
                ex = rec.get("execute_at")
                if ex is None or len(pxy) != len(tau) or not pxy:
                    continue
                ex_sec = ex.to_sec()
                shifted = [ex_sec + float(tau[j]) - t_ego_sec for j in range(len(tau))]
                op2, ot2 = self._downsample_path_xy_and_times(pxy, shifted)
                other_paths.append(op2)
                other_times_shifted.append(ot2)
            if not other_paths:
                times = self._solo_velocity_waypoint_times(ego_plan)
                self._simultaneous_optimized_times = [times]
                self._armed_waypoint_times_for_snapshot = list(times)
                self._pending_execute_at_for_arm = T_ego
                self._simultaneous_solve_done = True
                rospy.loginfo("MultiAgentNavigator: sequential thread fell back to solo (no valid peers)")
                return
            occ = self._build_sp_occupancy_grid()
            if occ is None:
                raise RuntimeError("occupancy grid missing for sequential planner")
            planner = MultiAgentSequentialPlanner(
                occ, other_paths, other_times_shifted, path=ego_path, v=float(self._multi_agent_max_velocity)
            )
            ego_times = planner.plan()
            if len(ego_times) != len(ego_plan):
                raise RuntimeError("sequential times length mismatch vs ego plan")
            self._simultaneous_optimized_times = [ego_times]
            self._armed_waypoint_times_for_snapshot = list(ego_times)
            self._pending_execute_at_for_arm = T_ego
            self._simultaneous_solve_done = True
            rospy.loginfo(
                "MultiAgentNavigator: sequential plan solved plan_id=%s T_ego=%s ego_wp=%d peers=%d",
                self._multi_plan_id,
                T_ego,
                len(ego_times),
                len(other_paths),
            )
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: sequential plan failed: %s", exc)
            self._simultaneous_solve_failed = True
        finally:
            self._simultaneous_solve_running = False

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
            fleet = [int(x) for x in self._multi_fleet_robot_ids]
            my_id = int(self.my_id)
            ego_k = fleet.index(my_id) if my_id in fleet else 0
            if optimized_times and ego_k < len(optimized_times):
                self._armed_waypoint_times_for_snapshot = [float(x) for x in optimized_times[ego_k]]
            if len(fleet) > 1:
                ts_msg = MultiAgentTimingSolve()
                ts_msg.plan_id = self._multi_plan_id
                ts_msg.source_agent = int(self.my_id)
                ts_msg.fleet_robot_ids = fleet
                ts_msg.waypoint_counts = [len(t) for t in optimized_times]
                flat = []
                for t in optimized_times:
                    flat.extend(float(x) for x in t)
                ts_msg.waypoint_times_flat = flat
                self._multi_agent_timing_solve_for_dds_pub.publish(ts_msg)
                rospy.loginfo(
                    "MultiAgentNavigator: published MultiAgentTimingSolve for DDS (%d agents) plan_id=%s",
                    len(fleet),
                    self._multi_plan_id,
                )
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
        hdr = path_msg.header
        plan = getattr(self, "unsmoothed_plan", None) or []
        for state in plan:
            pose_st = PoseStamped()
            pose_st.header = hdr
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
            and msg.plan_id == self._multi_plan_id
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
            self._sticky_multi_timing_replan = False
            self._replan_as_multi_from_external_goal_multi = False
            return

        self._peer_multi_planned_paths = {}
        self._reset_timing_compute_state()

        self._replan_as_multi_from_external_goal_multi = True
        self._from_multi_robot_goal = True
        self._multi_plan_id = msg.plan_id
        self._multi_coordinated = bool(msg.coordinated)
        self._multi_source_agent = int(msg.source_agent)
        self._multi_fleet_robot_ids = [int(x) for x in (msg.fleet_robot_ids or [])]
        if len(self._multi_fleet_robot_ids) == 1 and not (self._multi_plan_id or "").strip():
            self._multi_plan_id = "solo_multi"

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

        from_goal = self._from_multi_robot_goal
        sticky = self._sticky_multi_timing_replan
        multi = from_goal or sticky
        super(MultiAgentNavigator, self).replan(obj_x, obj_y, obj_d)

        if multi:
            self._from_multi_robot_goal = False
            fleet = [int(x) for x in self._multi_fleet_robot_ids]
            if sticky and not from_goal and len(fleet) >= 2:
                self._peer_multi_planned_paths = {}
                self._multi_fleet_robot_ids = [int(self.my_id)]
                fleet = [int(self.my_id)]
            elif len(fleet) >= 2:
                self._peer_multi_planned_paths = {}
            self._reset_timing_compute_state()
            if sticky and not from_goal:
                rospy.loginfo(
                    "MultiAgentNavigator: sticky replan plan_id=%s fleet=%s",
                    self._multi_plan_id or "?",
                    list(self._multi_fleet_robot_ids),
                )
            if self.mode in (l2.Mode.ALIGN, l2.Mode.TRACK, l2.Mode.PARK):
                plan = getattr(self, "unsmoothed_plan", None) or []
                if len(plan) >= 2:
                    dx = float(plan[1][0]) - float(plan[0][0])
                    dy = float(plan[1][1]) - float(plan[0][1])
                    self.th_init = float(np.arctan2(dy, dx))
                self.heading_controller.load_goal(self.th_init)
                self._awaiting_pre_multi_align = True
                self._pre_multi_align_started_at = rospy.Time.now()
                self.switch_mode(l2.Mode.ALIGN)
                rospy.loginfo(
                    "MultiAgentNavigator: pre-MULTI ALIGN dwell (plan_id=%s) policy=%s max_s=%.2f",
                    self._multi_plan_id,
                    self._pre_multi_align_exit_policy,
                    self._pre_multi_align_sec,
                )
            elif self.mode == l2.Mode.IDLE:
                self._multi_plan_id = ""
                self._multi_coordinated = False
                self._multi_source_agent = 0
                self._multi_fleet_robot_ids = []
                self._sticky_multi_timing_replan = False
                self._replan_as_multi_from_external_goal_multi = False
                self._reset_timing_compute_state()

    def publish_control(self):
        if (
            self._sticky_multi_timing_replan
            and self.mode == l2.Mode.IDLE
            and self.x_g is None
            and self.y_g is None
            and self.theta_g is None
        ):
            self._sticky_multi_timing_replan = False
            self._replan_as_multi_from_external_goal_multi = False
        if self._awaiting_pre_multi_align and self.mode != l2.Mode.ALIGN:
            rospy.logwarn_throttle(
                5.0,
                "MultiAgentNavigator: pre-MULTI align expected ALIGN but mode=%s; clearing pre-align flag",
                self.mode,
            )
            self._clear_pre_multi_align_state()

        if self._awaiting_pre_multi_align and self.mode == l2.Mode.ALIGN:
            if self._pre_multi_align_started_at is None:
                self._pre_multi_align_started_at = rospy.Time.now()
            elapsed = (rospy.Time.now() - self._pre_multi_align_started_at).to_sec()
            pol = self._pre_multi_align_exit_policy
            if pol == "max_only":
                exit_dwell = elapsed >= self._pre_multi_align_sec
            else:
                exit_dwell = self.aligned() or elapsed >= self._pre_multi_align_sec
            if exit_dwell:
                self._awaiting_pre_multi_align = False
                self.switch_mode(MultiagentComputingMode.MULTIAGENT_CONTROL_COMPUTING)
                rospy.loginfo(
                    "Entering MULTIAGENT_CONTROL_COMPUTING after pre-align (plan_id=%s)",
                    self._multi_plan_id,
                )
                self._multi_phase_started_at = rospy.Time.now()
                self._publish_planned_path_for_multi_dds()
            else:
                super(MultiAgentNavigator, self).publish_control()
                return

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
                        self._multi_plan_id or "?",
                        self._multi_fleet_robot_ids,
                    )
                    self._abort_multi_to_idle()
                    return
            if (
                self._fleet_paths_complete()
                and not self._simultaneous_solve_done
                and not self._simultaneous_solve_failed
                and not self._simultaneous_solve_running
            ):
                if self._timing_solve_wait_started_at is None:
                    self._timing_solve_wait_started_at = rospy.Time.now()
                elif (
                    rospy.Time.now() - self._timing_solve_wait_started_at
                ).to_sec() > self._multi_agent_timing_solve_wait_timeout_sec:
                    rospy.logwarn(
                        "MultiAgentNavigator: timing solve wait timeout (%.1fs) plan_id=%s fleet=%s -> IDLE",
                        self._multi_agent_timing_solve_wait_timeout_sec,
                        self._multi_plan_id or "?",
                        self._multi_fleet_robot_ids,
                    )
                    self._abort_multi_to_idle()
                    return
            else:
                self._timing_solve_wait_started_at = None
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
                self._multi_plan_id or "?",
                msg,
            )
            self.prev_om = 0.0
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            return
        super(MultiAgentNavigator, self).publish_control()

    def shutdown_callback(self):
        if self._active_traj_prune_timer is not None:
            self._active_traj_prune_timer.shutdown()
            self._active_traj_prune_timer = None
        super(MultiAgentNavigator, self).shutdown_callback()


if __name__ == "__main__":
    nav = MultiAgentNavigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(3)
    nav.run()
