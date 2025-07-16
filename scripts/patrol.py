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

        self.pose_pub = rospy.Publisher('/external_goal', Pose2D, queue_size=10)
        robot_state_subscriber = rospy.Subscriber('/robot_mode', Int32, self.robot_state_callback)
        localized_subscriber = rospy.Subscriber('/localized', Bool, self.localized_callback)

    def load_patrol_points(self):
        # Load patrol points from the /workspace/catkin_ws/src/mattbot_navigation/scripts/patrol.txt file
        patrol_points = []
        try:
            with open('/workspace/catkin_ws/src/mattbot_navigation/scripts/patrol.txt', 'r') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        x, y, theta = map(float, line.strip().split(','))
                        patrol_points.append(Pose2D(x=x, y=y, theta=theta))
        except FileNotFoundError:
            rospy.logerr("Patrol points file not found.")
        except ValueError:
            rospy.logerr("Error parsing patrol points file. Ensure the format is correct.")
        return patrol_points

    def robot_state_callback(self, msg):
        if msg.data == 5 and self.robot_state != 5:
            self.ready_for_next_point = True
            self.parking_time = rospy.get_time()
        
        self.robot_state = msg.data

    def localized_callback(self, msg):
        if msg.data and not self.localized:
            rospy.loginfo("Robot localized, ready to start patrol.")
            self.localized = True
            self.ready_for_next_point = True
            self.parking_time = rospy.get_time()

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if self.ready_for_next_point and (current_time - self.parking_time) > 10:
                self.publish_patrol_point()
            rate.sleep()

    def publish_patrol_point(self):
        self.pose_pub.publish(self.patrol_points[self.current_index])
        self.current_index += 1
        self.current_index %= len(self.patrol_points)
        self.ready_for_next_point = False
        rospy.loginfo(f"Published patrol point {self.current_index} at {self.patrol_points[self.current_index]}")


if __name__ == '__main__':
    patrol = Patrol()
    patrol.run()