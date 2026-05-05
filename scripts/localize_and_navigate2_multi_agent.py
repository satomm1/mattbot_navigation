#!/usr/bin/env python3
"""
Navigator variant: same behavior as localize_and_navigate2, plus /external_goal_multi
(MultiRobotExternalGoal). After a successful plan from a multi-robot goal, enters
MULTIAGENT_CONTROL_COMPUTING (integer 11 on /robot_mode) and holds until future
multi-agent control is implemented.

Python 3.8 does not allow subclassing an existing Enum with new members, so the extra
state uses a separate IntEnum with value 11 (distinct from localize_and_navigate2.Mode).
"""

import time
from enum import IntEnum

import rospy
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Path
from mattbot_dds.msg import MultiAgentPlannedPath, MultiRobotExternalGoal

import localize_and_navigate2 as l2


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
        super(MultiAgentNavigator, self).__init__(node_name=node_name)
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

    def _peer_multi_agent_planned_path_callback(self, msg):
        if int(msg.source_agent) == int(self.my_id):
            return
        self._peer_multi_planned_paths[int(msg.source_agent)] = msg
        rospy.logdebug_throttle(
            5.0,
            "MultiAgentNavigator: peer planned paths stored: %s",
            list(self._peer_multi_planned_paths.keys()),
        )

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

        self._from_multi_robot_goal = True
        self._multi_plan_id = msg.plan_id
        self._multi_coordinated = bool(msg.coordinated)
        self._multi_source_agent = int(msg.source_agent)

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
                self._publish_planned_path_for_multi_dds()
            elif self.mode == l2.Mode.IDLE:
                self._multi_plan_id = ""
                self._multi_coordinated = False
                self._multi_source_agent = 0

    def publish_control(self):
        if self._is_multiagent_computing_mode(self.mode):
            rospy.loginfo_throttle(
                5.0,
                "MULTIAGENT_CONTROL_COMPUTING (plan_id=%s); holding cmd_vel=0",
                getattr(self, "_multi_plan_id", "") or "?",
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
