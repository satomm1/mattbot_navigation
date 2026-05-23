#!/usr/bin/env python3
"""
Navigator variant: extends ``localize_and_navigate.Navigator`` with optional **coordinated timing**
after a geometric path exists. Geometric planning (A*, smoothing, ALIGN/TRACK/PARK) still lives in the
parent class; this file adds a **multi-agent timing phase** so several robots can agree on waypoint
times and a common ``execute_at`` wall clock before TRACK.

**When does multi timing run?**
    Only if ``replan()`` is entered with ``_from_multi_robot_goal`` (fresh Pose2D or ``/external_goal_multi``)
    or ``_sticky_multi_timing_replan`` (see below). Otherwise behavior matches the single-robot navigator.

**End-to-end lifecycle (happy path)**
    1. Goal arrives → ``external_goal_callback`` or ``external_goal_multi_callback`` sets fleet metadata
       and calls ``replan()`` with ``_from_multi_robot_goal`` True.
    2. Parent ``Navigator.replan()`` builds ``unsmoothed_plan`` / smoothed preview and sets mode ALIGN
       or TRACK (same as base).
    3. This subclass then forces **pre-MULTI ALIGN**: dwell in ALIGN so headings settle before DDS
       publishes the grid plan.
    4. Mode ``MULTIAGENT_CONTROL_COMPUTING`` (value 12): ego publishes ``MultiAgentPlannedPath`` to ROS
       (``dds_data_publisher`` forwards to DDS). Peers do the same for the same ``plan_id``.
    5. **Coordinator** (robot id ``min(fleet_robot_ids)``) runs **one** of:
       - **Simultaneous** (fleet size ≥ 2): ``MultiAgentSimultaneousPlanner`` MILP; publishes
         ``MultiAgentTimingSolve`` for non-coordinator fleet members.
       - **Sequential** (singleton fleet + peer ``MultiAgentActiveTrajectory`` in local cache):
         ``MultiAgentSequentialPlanner`` with time-shifted peer trajectories.
       - **Solo** (singleton, no usable peer cache): analytic cumulative times at ``max_velocity``.
    6. Pending spline is **armed**; robots wait for ``MultiAgentExecuteAt`` (or coordinator auto-publishes
       after ``~multi_agent_auto_execute_delay_sec``). At ``execute_at``, ``_commit_timed_trajectory()``
       loads the timed spline, republishes visualization, publishes **active trajectory** for peers,
       and switches to TRACK. Parent logic then tracks the trajectory as usual.

**Sticky replans** (``/external_goal_multi`` missions only)
    After the first successful timed TRACK commit, ``_sticky_multi_timing_replan`` is True so **timeout /
    waypoint-miss / path-invalid replans** from the parent ``run()`` loop re-enter steps 3–6 instead of
    doing only a local single-robot replan. If the original fleet had ≥ 2 robots, the sticky path **shrinks
    the local fleet to ego only** so timing uses sequential/solo vs peer **active trajectories**, not a
    second simultaneous MILP (peers may still be executing their old coordinated plan).

**ROS ↔ DDS**
    This node uses ROS topics only; ``mattbot_dds`` scripts bridge ``*_for_dds`` / ``*_from_agent`` topics
    to CycloneDDS. Message definitions live in ``mattbot_dds/msg``.

**Python 3.8 note**
    Parent ``Mode`` enum cannot be extended; extra mode uses ``MultiagentComputingMode`` value 12.
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
from navigation_utils import compute_trajectory_from_timed_waypoints, plan_start_heading
from social_path_planning.occupancy_grid import StochOccupancyGrid2D as SpStochOccupancyGrid2D
from visualization_msgs.msg import Marker, MarkerArray
from social_path_planning.multi_planning import MultiAgentSequentialPlanner, MultiAgentSimultaneousPlanner


def _load_localize_and_navigate():
    """
    Load the navigator implementation from scripts/localize_and_navigate.py.

    We cannot ``import localize_and_navigate`` when this file is run via the Catkin
    devel wrapper: that name resolves to another wrapper script in lib/pkg/ which execs
    the real file into a private dict, so the imported module has no ``Navigator``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "localize_and_navigate.py")
    if not os.path.isfile(path):
        try:
            import rospkg

            path = os.path.join(rospkg.RosPack().get_path("mattbot_navigation"), "scripts", "localize_and_navigate.py")
        except Exception:
            pass
    if not os.path.isfile(path):
        raise ImportError("Could not find localize_and_navigate.py (tried next to %r)" % (__file__,))
    spec = importlib.util.spec_from_file_location("_localize_and_navigate_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nav = _load_localize_and_navigate()


class MultiagentComputingMode(IntEnum):
    """Extra navigator mode; value 12 must not collide with ``localize_and_navigate.Mode`` (0–11).

    Used while the robot is stationary: waiting for peer planned paths, running MILP/sequential timing,
    arming the spline, and synchronizing on ``execute_at``.
    """

    MULTIAGENT_CONTROL_COMPUTING = 12


class MultiAgentNavigator(nav.Navigator):
    """Adds multi-robot **timing** (waypoint times + execute_at) on top of the base geometric navigator.

    Read the module docstring for the full lifecycle. Key instance state:

    - ``_from_multi_robot_goal``: set True immediately before ``replan()`` for a **new** multi-aware goal
      (Pose2D path uses singleton fleet; ``MultiRobotExternalGoal`` carries full fleet metadata).
    - ``_sticky_multi_timing_replan``: after first timed TRACK from ``/external_goal_multi``, keeps later
      replans inside the coordinated timing pipeline.
    - ``_multi_fleet_robot_ids`` / ``_multi_plan_id``: roster and correlation id for DDS messages.
    - ``_peer_multi_planned_paths``: latest ``MultiAgentPlannedPath`` per **source_agent** (needed for
      simultaneous MILP when fleet size ≥ 2).
    - ``_peer_active_trajectories``: cache of peers' **committed** timed plans (for sequential replanning).
    """

    def __init__(self, node_name="mattbot_navigator_multi_agent"):
        # --- Goal-source flags (drive whether replan() re-enters pre-MULTI + timing) ---
        self._from_multi_robot_goal = False
        self._replan_as_multi_from_external_goal_multi = False
        self._sticky_multi_timing_replan = False
        # --- Mission identity (from MultiRobotExternalGoal; Pose2D uses solo_pose + [my_id]) ---
        self._multi_plan_id = ""
        self._multi_coordinated = False
        self._multi_source_agent = 0
        self._peer_multi_planned_paths = {}
        self._multi_fleet_robot_ids = []
        # --- MULTI phase wall-clock (timeouts in publish_control) ---
        self._multi_phase_started_at = None
        # --- Coordinator-only MILP / sequential / solo worker (daemon threads) ---
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._simultaneous_lock = threading.Lock()
        # --- Arm / commit handoff: spline built in MULTI, executed at execute_at on TRACK ---
        self._timed_traj_armed = False
        self._pending_traj_times = None
        self._pending_traj = None
        self._execute_at_ros_time = None
        self._timed_arm_rostime = None
        self._auto_execute_timer = None
        # --- Pre-MULTI ALIGN dwell (heading settle before publishing planned path) ---
        self._awaiting_pre_multi_align = False
        self._pre_multi_align_started_at = None
        self._timing_solve_wait_started_at = None
        # --- Peer timed trajectories (ROS, may be bridged from DDS active_trajectory) ---
        self._peer_active_trajectories = {}
        self._peer_traj_cache_lock = threading.Lock()
        self._pending_execute_at_for_arm = None
        self._armed_waypoint_times_for_snapshot = None
        self._active_traj_prune_timer = None
        super(MultiAgentNavigator, self).__init__(node_name=node_name)
        # --- Parameters (see rospy param names in get_param calls) ---
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
        self._multi_agent_waypoint_time_buffer_sec = float(
            rospy.get_param("~multi_agent_waypoint_time_buffer_sec", 5.0)
        )
        self._multi_agent_plan_duration_slack_sec = float(
            rospy.get_param("~multi_agent_plan_duration_slack_sec", 5.0)
        )
        self._multi_agent_disable_person_and_peer_avoidance = bool(
            rospy.get_param("~multi_agent_disable_person_and_peer_avoidance", True)
        )
        self._leader_execute_dds_topic = rospy.get_param(
            "~multi_agent_execute_at_dds_trigger_topic", "/multi_agent_execute_at_dds"
        ).strip() or "/multi_agent_execute_at_dds"
        self._leader_execute_pub = rospy.Publisher(self._leader_execute_dds_topic, MultiAgentExecuteAt, queue_size=2, latch=False)
        # Fleet goals with plan_id + roster (often from DDS → own_data_subscriber → this topic).
        topic = rospy.get_param("~external_goal_multi_topic", "/external_goal_multi").strip() or "/external_goal_multi"
        rospy.Subscriber(topic, MultiRobotExternalGoal, self.external_goal_multi_callback, queue_size=10)
        rospy.logdebug("MultiAgentNavigator: subscribed to %s", topic)

        # Ego publishes grid path for DDS; peers' paths arrive on *_from_agent (via DDS → data_subscriber).
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
        rospy.logdebug(
            "MultiAgentNavigator: DDS planned path pub=%s sub=%s",
            self._dds_planned_pub_topic,
            self._peer_planned_sub_topic,
        )
        # execute_at: wall time when t=0 of the pending timed spline begins (fleet-wide agreement).
        self._execute_at_sub_topic = rospy.get_param("~multi_agent_execute_at_topic", "/multi_agent_execute_at").strip() or "/multi_agent_execute_at"
        rospy.Subscriber(self._execute_at_sub_topic, MultiAgentExecuteAt, self._multi_agent_execute_at_callback, queue_size=10)
        rospy.logdebug("MultiAgentNavigator: subscribed to execute_at %s", self._execute_at_sub_topic)
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
        rospy.logdebug(
            "MultiAgentNavigator: timing solve pub=%s sub=%s",
            self._timing_solve_for_dds_topic,
            self._timing_solve_sub_topic,
        )
        # Coordinator may publish execute_at locally after a delay so non-ROS peers still receive it via DDS.
        rospy.logdebug(
            "MultiAgentNavigator: auto_execute=%s coordinator_delay_s=%.2f (coordinator=min fleet_robot_ids)",
            self._multi_agent_auto_execute,
            self._multi_agent_auto_execute_delay_sec,
        )
        self._pre_multi_align_sec = float(rospy.get_param("~multi_agent_pre_multi_align_sec", 4.0))
        self._pre_multi_align_exit_policy = rospy.get_param(
            "~multi_agent_pre_multi_align_exit_policy", "aligned_or_max"
        ).strip().lower()
        if self._pre_multi_align_exit_policy not in ("aligned_or_max", "max_only"):
            rospy.logwarn(
                "MultiAgentNavigator: unknown ~multi_agent_pre_multi_align_exit_policy=%r; using aligned_or_max",
                self._pre_multi_align_exit_policy,
            )
            self._pre_multi_align_exit_policy = "aligned_or_max"
        rospy.logdebug(
            "MultiAgentNavigator: pre_multi_align max_s=%.2f exit_policy=%s",
            self._pre_multi_align_sec,
            self._pre_multi_align_exit_policy,
        )
        rospy.logdebug(
            "MultiAgentNavigator: coordinated MILP on min(fleet_robot_ids); timing via MultiAgentTimingSolve"
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
        rospy.logdebug(
            "MultiAgentNavigator: active trajectory pub=%s sub=%s budget_s=%.2f",
            self._active_traj_for_dds_topic,
            self._active_traj_sub_topic,
            self._multi_agent_sequential_budget_sec,
        )
        # Periodically drop stale peer trajectories so sequential planner does not use ancient geometry.
        self._active_traj_prune_timer = rospy.Timer(
            rospy.Duration(max(0.2, self._multi_agent_active_traj_check_period_sec)),
            self._peer_active_trajectory_prune_timer_cb,
        )
        rospy.loginfo(
            "MultiAgentNavigator: multi-agent ROS/DDS bridge ready "
            "(goal_multi=%s planned_pub=%s execute_at=%s timing_pub=%s active_pub=%s leader_exec=%s)",
            topic,
            self._dds_planned_pub_topic,
            self._execute_at_sub_topic,
            self._timing_solve_for_dds_topic,
            self._active_traj_for_dds_topic,
            self._leader_execute_dds_topic,
        )

    def external_goal_callback(self, msg):
        """Handle ``/external_goal`` and ``/voice_goal`` (geometry_msgs/Pose2D).

        Treats the goal as a **singleton fleet** (only this robot id): multi timing uses sequential
        planner if peer active trajectories exist in cache, otherwise solo analytic times. Does **not**
        set ``_replan_as_multi_from_external_goal_multi`` (no sticky replan pipeline for Pose2D goals).

        Matches base Navigator policy: ignore duplicate goals, refuse invalid cells, stop then IDLE if
        busy, then clear DDS path cache and call ``replan()`` with ``_from_multi_robot_goal`` True.
        """
        if (
            self.x_g is not None
            and self.y_g is not None
            and self.theta_g is not None
            and (msg.x == self.x_g and msg.y == self.y_g and msg.theta == self.theta_g)
        ):
            rospy.logdebug("External goal is the same as current goal, ignoring")
            return

        if self.mode == nav.Mode.WAITING_FOR_INIT or self.mode == nav.Mode.LOCALIZING:
            return

        if self.mode != nav.Mode.IDLE:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            self.switch_mode(nav.Mode.IDLE)

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
        if self._suppress_person_and_peer_avoidance():
            rospy.logdebug(
                "MultiAgentNavigator: person/peer avoidance disabled for plan_id=%s",
                self._multi_plan_id,
            )
        self.replan()

    def _clear_pre_multi_align_state(self):
        """Cancel the pre-MULTI ALIGN dwell (e.g. on abort or after entering MULTI)."""
        self._awaiting_pre_multi_align = False
        self._pre_multi_align_started_at = None

    def aligned(self):
        """Block parent ``run()`` ALIGN→TRACK while pre-MULTI dwell is active (see ``publish_control``)."""
        if self._awaiting_pre_multi_align:
            return False
        return super(MultiAgentNavigator, self).aligned()

    def _suppress_person_and_peer_avoidance(self):
        """While a multi-agent mission is active, follow the timed plan; do not stop/slow for people or replan around peers in A*."""
        if not (self._multi_plan_id or "").strip():
            return False
        return bool(self._multi_agent_disable_person_and_peer_avoidance)

    def modify_velocity_for_person(self, V, om):
        if self._suppress_person_and_peer_avoidance():
            return V, om
        return super(MultiAgentNavigator, self).modify_velocity_for_person(V, om)

    def agent_intersect_path(self):
        if self._suppress_person_and_peer_avoidance():
            self.agents_in_path = set()
            return False
        return super(MultiAgentNavigator, self).agent_intersect_path()

    def _reset_timing_compute_state(self):
        """Reset MILP/sequential/solo worker flags and pending arm state between goals or replan cycles.

        Does **not** clear ``_multi_plan_id`` / fleet roster; callers do that when abandoning the mission.
        """
        self._simultaneous_solve_done = False
        self._simultaneous_solve_failed = False
        self._simultaneous_solve_running = False
        self._simultaneous_optimized_times = None
        self._multi_phase_started_at = None
        self._reset_timed_execute_state()
        self._clear_pre_multi_align_state()
        self._timing_solve_wait_started_at = None

    def _fleet_coordinator_id(self):
        """Return ``min(_multi_fleet_robot_ids)`` or None if fleet empty.

        The lowest robot id runs the timing optimization and publishes ``MultiAgentTimingSolve`` /
        optional ``MultiAgentExecuteAt`` for the fleet.
        """
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        return min(fleet) if fleet else None

    def _is_solo_fleet_timing(self):
        """True when timing is solo/sequential (no simultaneous MILP); track start is anchored locally."""
        return len(self._multi_fleet_robot_ids) <= 1

    def _solo_scheduled_execute_at(self):
        """Wall time when solo/sequential TRACK should begin (small lead for controller settle)."""
        return rospy.Time.now() + rospy.Duration(max(0.05, self._multi_agent_auto_execute_wall_extra_sec))

    def _i_am_coordinator(self):
        """True if this robot is the designated coordinator for the current fleet roster."""
        cid = self._fleet_coordinator_id()
        return cid is not None and int(self.my_id) == cid

    @staticmethod
    def _unpack_timing_solve_flat(counts, flat):
        """Unpack ``MultiAgentTimingSolve.waypoint_times_flat`` using ``waypoint_counts`` per robot."""
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
        """Convert ``nav_msgs/Path`` to ``[(x,y), ...]`` for planners and cache records."""
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
        """True if cache holds at least one **other** robot with path + times suitable for sequential MILP."""
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
        """Ingest ``MultiAgentActiveTrajectory`` from ROS (often bridged from DDS).

        Stores polyline + cumulative waypoint times + execute_at so ``MultiAgentSequentialPlanner`` can
        shift peers into ego's time base. Inactive messages remove that robot from the cache.
        """
        if int(msg.robot_id) == int(self.my_id):
            return
        if not msg.active and int(msg.robot_id) in self._peer_active_trajectories:
            with self._peer_traj_cache_lock:
                self._peer_active_trajectories.pop(int(msg.robot_id), None)
            rospy.logdebug("MultiAgentNavigator: peer %s active trajectory cleared (inactive)", msg.robot_id)
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
        rospy.logdebug(
            "MultiAgentNavigator: cached active trajectory robot=%s wp=%d plan_id=%s",
            msg.robot_id,
            len(path_xy),
            msg.plan_id,
        )

    def _peer_active_trajectory_prune_timer_cb(self, _evt=None):
        """Drop peer cache entries that are too old, inactive, or clearly finished in wall time."""
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
        """Robot ids to embed in ``MultiAgentActiveTrajectory.dds_forward_robot_ids`` for DDS fan-out.

        Includes configured extras, current fleet mates (except self), and anyone present in the
        peer trajectory cache.
        """
        ids = set(int(x) for x in self._multi_agent_active_traj_forward_extra_ids)
        with self._peer_traj_cache_lock:
            ids.update(int(k) for k in self._peer_active_trajectories.keys())
        for rid in self._multi_fleet_robot_ids:
            ids.add(int(rid))
        ids.discard(int(self.my_id))
        return sorted(ids)

    def _publish_active_trajectory_msg(self, active, path_msg, waypoint_times, execute_at, plan_id):
        """Publish our timed plan snapshot so peers can treat us as a moving obstacle (sequential mode)."""
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
        """Cancel one-shot coordinator timer for delayed ``execute_at`` publish."""
        if self._auto_execute_timer is not None:
            self._auto_execute_timer.shutdown()
            self._auto_execute_timer = None

    def _reset_timed_execute_state(self):
        """Clear arm/commit fields (pending spline, execute_at, auto timer). Called when leaving MULTI."""
        self._cancel_auto_execute_timer()
        self._timed_traj_armed = False
        self._pending_traj_times = None
        self._pending_traj = None
        self._execute_at_ros_time = None
        self._timed_arm_rostime = None

    def _multi_agent_execute_at_callback(self, msg):
        """Fill ``_execute_at_ros_time`` when DDS/ROS delivers ``MultiAgentExecuteAt`` for our plan_id."""
        if not self._timed_traj_armed:
            return
        if (msg.plan_id or "").strip() != (self._multi_plan_id or "").strip():
            return
        if self._is_solo_fleet_timing():
            return
        self._execute_at_ros_time = msg.execute_at
        rospy.logdebug(
            "MultiAgentNavigator: execute_at received plan_id=%s execute_at=%s",
            msg.plan_id,
            self._execute_at_ros_time,
        )

    def _multi_agent_timing_solve_callback(self, msg):
        """Non-coordinator fleet members: apply ``MultiAgentTimingSolve`` from coordinator over ROS/DDS.

        Validates plan_id, coordinator id, fleet roster, and per-robot waypoint counts vs flat buffer.
        Coordinator ignores its own DDS echo (already applied inside ``_simultaneous_planner_thread_main``).
        """
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
        # Coordinator: empty payload = simultaneous MILP infeasible / fleet-wide abort (peers leave MULTI).
        if len(counts) == 0 and len(flat) == 0:
            rospy.logwarn(
                "MultiAgentNavigator: coordinated timing aborted (infeasible or solver failure) plan_id=%s",
                pid,
            )
            self._simultaneous_solve_failed = True
            return

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
        rospy.logdebug(
            "MultiAgentNavigator: applied timing_solve from coordinator=%s plan_id=%s ego_wp=%d",
            msg.source_agent,
            pid,
            len(rows[ego_k]),
        )

    def _maybe_arm_timed_trajectory(self):
        """After timing solve: build spline from ``unsmoothed_plan`` + per-waypoint times; arm for execute_at.

        On success sets ``_timed_traj_armed`` and either uses ``_pending_execute_at_for_arm`` from the
        worker thread or schedules coordinator auto-publish. Short paths fall back to PARK and clear
        multi session flags.
        """
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
            rospy.loginfo("MultiAgentNavigator: timed path too short; PARK_POSE to goal")
            self._set_park_goal_from_traj(self._traj_endpoint_for_park(plan))
            self._enter_park_pose()
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
        if self._is_solo_fleet_timing():
            # Solo/sequential: do not use execute_at stamped at MILP/solve time (stale vs commit → waypoint miss).
            self._pending_execute_at_for_arm = None
            self._execute_at_ros_time = self._solo_scheduled_execute_at()
        elif self._pending_execute_at_for_arm is not None:
            self._execute_at_ros_time = self._pending_execute_at_for_arm
            self._pending_execute_at_for_arm = None
        else:
            self._execute_at_ros_time = None
            self._schedule_leader_auto_execute()
        rospy.logdebug(
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
        rospy.logdebug(
            "MultiAgentNavigator: coordinator robot %s scheduling auto execute_at publish in %.2fs -> %s",
            coord,
            d,
            self._leader_execute_dds_topic,
        )
        self._auto_execute_timer = rospy.Timer(rospy.Duration(d), self._leader_auto_execute_timer_cb, oneshot=True)

    def _leader_auto_execute_timer_cb(self, _evt=None):
        """One-shot: coordinator publishes ``MultiAgentExecuteAt`` to the DDS trigger topic after delay."""
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
        rospy.logdebug(
            "MultiAgentNavigator: coordinator published MultiAgentExecuteAt plan_id=%s execute_at=%s (DDS trigger %s)",
            pid,
            msg.execute_at,
            self._leader_execute_dds_topic,
        )

    def _commit_timed_trajectory(self):
        """Apply the armed timed spline at ``_execute_at_ros_time`` and transition to TRACK.

        Copies trajectory into parent ``current_plan*`` fields, rebuilds waypoint markers for the parent
        TRACK watchdog, publishes ``MultiAgentActiveTrajectory`` for peers, clears MULTI-only solver
        flags, and enables **sticky replans** when this mission came from ``/external_goal_multi``.
        """
        if not self._timed_traj_armed:
            return
        if self._pending_traj is None or self._pending_traj_times is None:
            rospy.logwarn("MultiAgentNavigator: commit called with no pending trajectory")
            self._abort_multi_to_idle()
            return
        if self._execute_at_ros_time is None:
            return
        now = rospy.Time.now()
        scheduled = self._execute_at_ros_time
        late_s = (now - scheduled).to_sec() if scheduled is not None else 0.0
        if (
            not self._is_solo_fleet_timing()
            and late_s > self._multi_agent_execute_max_lateness
            and self._multi_agent_execute_late_policy == "idle"
        ):
            rospy.logwarn(
                "MultiAgentNavigator: execute_at late by %.2fs (max %.2fs, policy=idle) -> IDLE",
                late_s,
                self._multi_agent_execute_max_lateness,
            )
            self._abort_multi_to_idle()
            return
        # TRACK t=0 and parent out-of-time watchdog use actual commit time (not fleet execute_at).
        track_start = now
        t_new = self._pending_traj_times
        traj_new = self._pending_traj
        planned_path = getattr(self, "unsmoothed_plan", None) or []
        dur_nom = float(t_new[-1])
        dur_slack = max(0.0, self._multi_agent_plan_duration_slack_sec)

        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub, times=t_new)
        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)
        self.current_plan = traj_new
        self.current_plan_start_time = track_start
        self.current_plan_duration = dur_nom + dur_slack
        rospy.loginfo(
            "MultiAgentNavigator: timed TRACK (nominal=%.2fs slack=%.2fs timeout=%.2fs execute_at_late=%.2fs)",
            dur_nom,
            dur_slack,
            self.current_plan_duration,
            late_s,
        )

        self._set_park_goal_from_traj(traj_new)

        self.th_init = plan_start_heading(planned_path, traj_new, v_min=0.05)
        self.heading_controller.load_goal(self.th_init)

        # Parent TRACK: deadline = plan_start + t_new[i] + buffer (avoid BACKING/replan when slightly late).
        wp_buf = max(0.0, self._multi_agent_waypoint_time_buffer_sec)
        self.waypoints = []
        marker_arr = MarkerArray()
        for i in range(20, len(traj_new), 20):
            self.waypoints.append([traj_new[i, 0], traj_new[i, 1], t_new[i] + wp_buf])
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
        self._publish_active_trajectory_msg(True, path_snap, wt_pub, scheduled, self._multi_plan_id)

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
        self.switch_mode(nav.Mode.TRACK)
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
        """Optional vertex cap for MILP (``~multi_agent_max_path_points``); no-op if param ≤ 0."""
        m = self._multi_max_path_points
        if m <= 0 or len(pts) <= m:
            return pts
        idx = np.unique(np.linspace(0, len(pts) - 1, num=m, dtype=int))
        return [pts[i] for i in idx]

    def _downsample_path_xy_and_times(self, pts, times):
        """Downsample path and parallel time arrays together (keeps indices aligned)."""
        m = self._multi_max_path_points
        if m <= 0 or len(pts) <= m or len(pts) != len(times):
            return pts, times
        idx = np.unique(np.linspace(0, len(pts) - 1, num=m, dtype=int))
        return [pts[i] for i in idx], [times[i] for i in idx]

    def _assemble_paths_for_simultaneous(self):
        """Build list of polylines (ego from ``unsmoothed_plan``, peers from ``_peer_multi_planned_paths``)."""
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
        """True when ego has a non-empty plan and every **other** fleet member has a matching ``plan_id`` path."""
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

    def _publish_timing_solve_abort_for_fleet(self):
        """Coordinator only: broadcast empty timing payload so fleet peers leave MULTI (infeasible MILP, etc.)."""
        if not self._i_am_coordinator():
            return
        fleet = [int(x) for x in self._multi_fleet_robot_ids]
        if len(fleet) < 2:
            return
        pid = (self._multi_plan_id or "").strip()
        if not pid:
            return
        ts_msg = MultiAgentTimingSolve()
        ts_msg.plan_id = pid
        ts_msg.source_agent = int(self.my_id)
        ts_msg.fleet_robot_ids = fleet
        ts_msg.waypoint_counts = []
        ts_msg.waypoint_times_flat = []
        self._multi_agent_timing_solve_for_dds_pub.publish(ts_msg)
        rospy.logwarn(
            "MultiAgentNavigator: published timing ABORT for fleet (peers -> IDLE) plan_id=%s fleet=%s",
            pid,
            fleet,
        )

    def _abort_multi_to_idle(self):
        """Hard stop MULTI: notify peers (inactive trajectory), clear mission id, reset timing state, IDLE."""
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
        self.switch_mode(nav.Mode.IDLE)

    def _try_simultaneous_plan_if_ready(self):
        """Coordinator-only entry: start exactly one timing worker when paths are ready.

        Branches: simultaneous (|fleet|≥2), sequential (singleton + peer cache), else solo analytic.
        Non-coordinators stay in MULTI until they receive ``MultiAgentTimingSolve`` via callback.
        """
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
        branch = (
            "simultaneous"
            if len(fleet) >= 2
            else ("sequential" if self._peer_active_obstacles_available() else "solo")
        )
        rospy.loginfo(
            "MultiAgentNavigator: timing solve plan_id=%s fleet=%s -> %s",
            self._multi_plan_id,
            fleet,
            branch,
        )
        if len(fleet) >= 2:
            threading.Thread(target=self._simultaneous_planner_thread_main, daemon=True).start()
        elif len(fleet) == 1 and self._peer_active_obstacles_available():
            threading.Thread(target=self._sequential_planner_thread_main, daemon=True).start()
        else:
            threading.Thread(target=self._solo_timing_thread_main, daemon=True).start()

    def _solo_timing_thread_main(self):
        """Background: analytic times only; sets ``_simultaneous_optimized_times`` as a one-row list."""
        try:
            plan = getattr(self, "unsmoothed_plan", None) or []
            if len(plan) < 1:
                raise RuntimeError("empty plan")
            times = self._solo_velocity_waypoint_times(plan)
            self._simultaneous_optimized_times = [times]
            self._armed_waypoint_times_for_snapshot = list(times)
            rospy.logdebug(
                "MultiAgentNavigator: solo timing ready plan_id=%s duration_s=%.3f (execute_at set at arm)",
                self._multi_plan_id,
                float(times[-1]) if times else 0.0,
            )
            self._simultaneous_solve_done = True
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: solo timing failed: %s", exc)
            self._simultaneous_solve_failed = True
        finally:
            self._simultaneous_solve_running = False

    def _sequential_planner_thread_main(self):
        """Background: ``MultiAgentSequentialPlanner`` using cached peer active trajectories (time-shifted).

        If no valid peer geometry is available, falls back to the same analytic times as solo mode.
        """
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
                self._simultaneous_solve_done = True
                rospy.logdebug("MultiAgentNavigator: sequential thread fell back to solo (no valid peers)")
                return
            if len(ego_path) < 2:
                times = self._solo_velocity_waypoint_times(ego_plan)
                self._simultaneous_optimized_times = [times]
                self._armed_waypoint_times_for_snapshot = list(times)
                self._simultaneous_solve_done = True
                rospy.logwarn(
                    "MultiAgentNavigator: sequential planner skipped (ego path has %d points); using solo analytic times",
                    len(ego_path),
                )
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
            self._simultaneous_solve_done = True
            rospy.logdebug(
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
        """Background: ``MultiAgentSimultaneousPlanner`` MILP; coordinator publishes ``MultiAgentTimingSolve``."""
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
            rospy.logdebug(
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
                rospy.logdebug(
                    "MultiAgentNavigator: published MultiAgentTimingSolve for DDS (%d agents) plan_id=%s",
                    len(fleet),
                    self._multi_plan_id,
                )
        except Exception as exc:
            rospy.logerr("MultiAgentNavigator: simultaneous plan failed: %s", exc)
            try:
                self._publish_timing_solve_abort_for_fleet()
            except Exception as pub_exc:
                rospy.logwarn("MultiAgentNavigator: fleet timing abort publish failed: %s", pub_exc)
            self._simultaneous_solve_failed = True
        finally:
            self._simultaneous_solve_running = False

    @staticmethod
    def _is_multiagent_computing_mode(mode):
        """True if ``mode`` is ``MultiagentComputingMode.MULTIAGENT_CONTROL_COMPUTING`` (value 12)."""
        return isinstance(mode, MultiagentComputingMode)

    def _path_from_unsmoothed_plan(self):
        """Build ``nav_msgs/Path`` from ``self.unsmoothed_plan`` rows ``[x, y, ...]`` (map frame)."""
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
        """Publish ego grid path for this ``plan_id`` so peers (and DDS) can run simultaneous timing."""
        path_msg = self._path_from_unsmoothed_plan()
        if not path_msg.poses:
            rospy.logwarn("MultiAgentNavigator: unsmoothed plan empty; skipping DDS planned path publish")
            return
        out = MultiAgentPlannedPath()
        out.plan_id = self._multi_plan_id
        out.source_agent = int(self.my_id)
        out.path = path_msg
        self._multi_agent_planned_path_pub.publish(out)
        rospy.logdebug(
            "MultiAgentNavigator: published MultiAgentPlannedPath for DDS (%d poses) plan_id=%s",
            len(path_msg.poses),
            self._multi_plan_id,
        )
        self._try_simultaneous_plan_if_ready()

    def _peer_multi_agent_planned_path_callback(self, msg):
        """Cache a peer's ``MultiAgentPlannedPath`` when ``plan_id`` matches our active mission."""
        if int(msg.source_agent) == int(self.my_id):
            return
        pid = (self._multi_plan_id or "").strip()
        if not pid or (msg.plan_id or "").strip() != pid:
            return
        self._peer_multi_planned_paths[int(msg.source_agent)] = msg
        n_poses = len(msg.path.poses)
        rospy.logdebug(
            "MultiAgentNavigator: received peer planned path source_agent=%d plan_id=%s poses=%d",
            int(msg.source_agent),
            msg.plan_id,
            n_poses,
        )
        self._try_simultaneous_plan_if_ready()

    def external_goal_multi_callback(self, msg):
        """Handle ``MultiRobotExternalGoal``: fleet roster + ``plan_id`` + goal pose (typically from DDS).

        Sets ``_replan_as_multi_from_external_goal_multi`` so the first successful timed TRACK enables
        **sticky** replans. Invalid cells clear sticky/session flags; duplicate pose+plan_id is ignored.
        """
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
            rospy.logdebug("Multi external goal unchanged (pose + plan_id), ignoring")
            return

        if self.mode == nav.Mode.WAITING_FOR_INIT or self.mode == nav.Mode.LOCALIZING:
            return

        if self.mode != nav.Mode.IDLE:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            self.switch_mode(nav.Mode.IDLE)

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
        if self._suppress_person_and_peer_avoidance():
            rospy.logdebug(
                "MultiAgentNavigator: person/peer avoidance disabled for plan_id=%s",
                self._multi_plan_id,
            )
        self.replan()

    def replan(self, obj_x=None, obj_y=None, obj_d=None):
        """Extend base ``replan()`` with optional **multi timing tail** (pre-MULTI ALIGN → MULTI → arm).

        ``Navigator.replan`` refuses to run while TRACK; parent ``run()`` switches IDLE first for
        timeout / invalid path replans. While ``mode == MULTIAGENT_CONTROL_COMPUTING`` this override
        **returns early** (replan is ignored until MULTI completes or aborts).

        Flags:
            * ``from_goal``: first replan after external callback set ``_from_multi_robot_goal``.
            * ``sticky``: subsequent replans for ``/external_goal_multi`` missions after first commit.

        Sticky + former multi-fleet (≥2): shrink ``_multi_fleet_robot_ids`` to ``[my_id]`` so timing uses
        sequential/solo vs peer active trajectories instead of a second simultaneous MILP.
        """
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
        if self._suppress_person_and_peer_avoidance():
            self.agents_in_path = set()
            self.other_agents_static = []
        super(MultiAgentNavigator, self).replan(obj_x, obj_y, obj_d)

        if multi:
            self._from_multi_robot_goal = False
            fleet = [int(x) for x in self._multi_fleet_robot_ids]
            # Sticky replan after a true multi-fleet mission: avoid a second simultaneous MILP on ego only.
            if sticky and not from_goal and len(fleet) >= 2:
                self._peer_multi_planned_paths = {}
                self._multi_fleet_robot_ids = [int(self.my_id)]
                fleet = [int(self.my_id)]
            elif len(fleet) >= 2:
                # First coordinated solve: wait for fresh peer MultiAgentPlannedPath messages.
                self._peer_multi_planned_paths = {}
            self._reset_timing_compute_state()
            if sticky and not from_goal:
                rospy.loginfo(
                    "MultiAgentNavigator: sticky replan plan_id=%s fleet=%s",
                    self._multi_plan_id or "?",
                    list(self._multi_fleet_robot_ids),
                )
            # Re-enter dwell + MULTI only when geometric replan produced a trackable plan.
            if self.mode in (nav.Mode.ALIGN, nav.Mode.TRACK, nav.Mode.PARK_POSE, nav.Mode.PARK_HEADING):
                traj = getattr(self, "current_plan", None)
                plan = getattr(self, "unsmoothed_plan", None) or []
                self.th_init = plan_start_heading(plan, traj, v_min=0.05)
                self.heading_controller.load_goal(self.th_init)
                self._awaiting_pre_multi_align = True
                self._pre_multi_align_started_at = rospy.Time.now()
                # Parent replan() may already have switched IDLE->ALIGN for heading; avoid ALIGN->ALIGN log/noise.
                if self.mode != nav.Mode.ALIGN:
                    self.switch_mode(nav.Mode.ALIGN)
                rospy.logdebug(
                    "MultiAgentNavigator: pre-MULTI ALIGN dwell (plan_id=%s) policy=%s max_s=%.2f",
                    self._multi_plan_id,
                    self._pre_multi_align_exit_policy,
                    self._pre_multi_align_sec,
                )
            elif self.mode == nav.Mode.IDLE:
                # Parent planning failed: tear down multi mission state.
                self._multi_plan_id = ""
                self._multi_coordinated = False
                self._multi_source_agent = 0
                self._multi_fleet_robot_ids = []
                self._sticky_multi_timing_replan = False
                self._replan_as_multi_from_external_goal_multi = False
                self._reset_timing_compute_state()

    def publish_control(self):
        """Drive pre-MULTI ALIGN dwell, MULTI timing phase, then defer to base controllers.

        Order matters:
            1) Clear sticky session if mission ended (IDLE + no goal).
            2) Pre-MULTI ALIGN: optionally block in ALIGN until dwell policy satisfied.
            3) MULTI: hold cmd_vel zero, enforce timeouts, run coordinator timing, arm, wait execute_at,
               commit to TRACK when wall clock allows.
            4) Otherwise ``Navigator.publish_control`` (ALIGN/TRACK/PARK/...).
        """
        # Goal reached: parent clears x_g/y_g/theta_g in PARK; drop sticky so next mission starts clean.
        if (
            self._sticky_multi_timing_replan
            and self.mode == nav.Mode.IDLE
            and self.x_g is None
            and self.y_g is None
            and self.theta_g is None
        ):
            self._sticky_multi_timing_replan = False
            self._replan_as_multi_from_external_goal_multi = False
        if self._awaiting_pre_multi_align and self.mode != nav.Mode.ALIGN:
            rospy.logwarn_throttle(
                5.0,
                "MultiAgentNavigator: pre-MULTI align expected ALIGN but mode=%s; forcing ALIGN",
                self.mode,
            )
            self.switch_mode(nav.Mode.ALIGN)
            super(MultiAgentNavigator, self).publish_control()
            return

        # Pre-MULTI ALIGN: stay in ALIGN until heading policy + max dwell satisfied, then enter MULTI.
        if self._awaiting_pre_multi_align and self.mode == nav.Mode.ALIGN:
            if self._pre_multi_align_started_at is None:
                self._pre_multi_align_started_at = rospy.Time.now()
            elapsed = (rospy.Time.now() - self._pre_multi_align_started_at).to_sec()
            pol = self._pre_multi_align_exit_policy
            if pol == "max_only":
                exit_dwell = elapsed >= self._pre_multi_align_sec
            else:
                exit_dwell = super(MultiAgentNavigator, self).aligned() or elapsed >= self._pre_multi_align_sec
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

        # MULTI phase: no base controller yet; coordinator solves timing; all robots wait execute_at.
        if self._is_multiagent_computing_mode(self.mode):
            if self._simultaneous_solve_failed:
                self._abort_multi_to_idle()
                return
            # Wait for peer MultiAgentPlannedPath messages (simultaneous fleet) before starting MILP.
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
            # Paths ready but non-coordinator still needs MultiAgentTimingSolve from DDS/ROS.
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
            rospy.logdebug_throttle(
                15.0,
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
        # Normal single-robot modes (from parent Navigator).
        super(MultiAgentNavigator, self).publish_control()

    def shutdown_callback(self):
        """Stop peer-cache timer then delegate to parent (zero cmd_vel on shutdown)."""
        if self._active_traj_prune_timer is not None:
            self._active_traj_prune_timer.shutdown()
            self._active_traj_prune_timer = None
        super(MultiAgentNavigator, self).shutdown_callback()


if __name__ == "__main__":
    # Same startup pattern as localize_and_navigate.py: brief delay for TF/subscribers, then parent's run loop.
    nav = MultiAgentNavigator()
    rospy.on_shutdown(nav.shutdown_callback)
    time.sleep(3)
    nav.run()
