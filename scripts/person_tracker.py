import rospy
import numpy as np
from sort import *

from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray

class PersonTracker:

    def __init__(self):
        self.tracker = Sort()

        self.num_detected = 0
        self.object_detect_subscriber = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=10)

    def detected_objects_callback(self, msg):
        object_array = msg.objects

        detections = []
        for obj in object_array:
            if obj.class_name == "person":
                detections.append([obj.x1, obj.y1, obj.x2, obj.y2, obj.probability])

        detections = np.array(detections)
        track_bbs_ids = self.tracker.update(detections)

        for track in track_bbs_ids:
            print(track)

        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.header.stamp = rospy.Time.now()
        # marker.ns = "person_tracker"
        # marker.id = s
        # marker.type = Marker.LINE_STRIP

        # for track in trackers:
        #     detected_object = DetectedObject()
        #     detected_object.x = track[0]
        #     detected_object.y = track[1]
        #     detected_object.width = track[2] - track[0]
        #     detected_object.height = track[3] - track[1]
        #     detected_object.confidence = track[4]
        #     detected_object.label = 'person'
        #     detected_object_array = DetectedObjectArray()
        #     detected_object_array.detected_objects.append(detected_object)
        #     self.person_tracker_publisher.publish(detected_object_array)

        # print(track_bbs_ids)



if __name__ == '__main__':
    rospy.init_node('person_tracker', anonymous=True)
    person_tracker = PersonTracker()
    rospy.spin()