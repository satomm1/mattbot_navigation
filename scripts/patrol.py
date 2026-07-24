import rospy
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32, Bool

class Patrol:

    def __init__(self):
        rospy.init_node('patrol_node', anonymous=True)
        self.patrol_points = self.load_patrol_points()

        self.current_index = 0
        self.robot_state = 0
        self.parking_time = 0
        self.ready_for_next_point = False
        self.localized = False
        # Wait before publishing the next patrol point.
        self.next_wait_sec = 0.0

        self.pose_pub = rospy.Publisher('/external_goal', Pose2D, queue_size=10)
        robot_state_subscriber = rospy.Subscriber('/robot_mode', Int32, self.robot_state_callback)
        localized_subscriber = rospy.Subscriber('/localized', Bool, self.localized_callback)

    def load_patrol_points(self):
        # Each row: x, y, theta, wait_sec (seconds to wait after reaching this point)
        patrol_points = []
        try:
            with open('/workspace/catkin_ws/src/mattbot_navigation/scripts/patrol.txt', 'r') as file:
                for line_num, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.strip().split(',')]
                    if len(parts) != 4:
                        raise ValueError(
                            f"line {line_num}: expected 4 values (x, y, theta, wait_sec), got {len(parts)}"
                        )
                    x, y, theta, wait_sec = map(float, parts)
                    patrol_points.append((Pose2D(x=x, y=y, theta=theta), wait_sec))
        except FileNotFoundError:
            rospy.logerr("Patrol points file not found.")
        except ValueError as e:
            rospy.logerr("Error parsing patrol points file: %s", e)
        return patrol_points

    def robot_state_callback(self, msg):
        if msg.data == 5 and self.robot_state != 5:
            self.ready_for_next_point = True
            self.parking_time = rospy.get_time()
        
        self.robot_state = msg.data

    def localized_callback(self, msg):
        if msg.data and not self.localized:
            rospy.loginfo("Robot localized; starting patrol in 10 seconds.")
            self.localized = True
            self.ready_for_next_point = True
            self.next_wait_sec = 10.0
            self.parking_time = rospy.get_time()

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if (
                self.ready_for_next_point
                and (current_time - self.parking_time) > self.next_wait_sec
            ):
                self.publish_patrol_point()
            rate.sleep()

    def publish_patrol_point(self):
        pose, wait_sec = self.patrol_points[self.current_index]
        self.pose_pub.publish(pose)
        self.next_wait_sec = wait_sec
        rospy.loginfo(
            "Published patrol point %d at (%.2f, %.2f, %.2f); wait after arrival: %.1fs",
            self.current_index, pose.x, pose.y, pose.theta, wait_sec,
        )
        self.current_index = (self.current_index + 1) % len(self.patrol_points)
        self.ready_for_next_point = False


if __name__ == '__main__':
    patrol = Patrol()
    patrol.run()
