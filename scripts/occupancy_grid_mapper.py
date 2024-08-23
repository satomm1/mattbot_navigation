import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import tf
import sensor_msgs.point_cloud2 as pc2
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray

import time

import numpy as np
from matplotlib import pyplot as plt

class StochOccupancyGrid2D(object):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                window_size, probs, thresh=0.5, robot_d=0.6):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.l = np.zeros((height, width))
        self.window_size = window_size # window_size
        # print(window_size)
        self.thresh = thresh
        self.robot_d=robot_d

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def snap_to_grid1(self, x):
        return (self.resolution * np.round(x[0] / self.resolution), self.resolution * np.round(x[1] / self.resolution))

    def get_index(self, x):
        return (np.round((x[0]-self.origin_x)/self.resolution), np.round((x[1]-self.origin_y)/self.resolution))

    def recalculate_probs(self):
        self.probs = 1 - 1/(1+np.exp(self.l))

class Map:

    def __init__(self, camera_height=0.216, total_height=0.5, robot_d=0.6):

        self.camera_height = camera_height
        self.total_height = total_height
        self.robot_d = robot_d

        # Tuning parameters for the inverse range sensor model
        self.alpha = 0.2   # 0.1 meters
        self.beta = 0.035  # 2 radians

        self.trans_listener = tf.TransformListener()

        self.is_localized = False
        self.localized_sub = rospy.Subscriber("/localized", Bool, self.localized_callback)

        # Get current map created from LIDAR SLAM
        self.map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        self.map_mod_msg = rospy.wait_for_message("/map_mod", OccupancyGrid)
        self.new_map = OccupancyGrid()
        self.new_map.header = self.map_msg.header
        self.new_map.info = self.map_msg.info
        self.new_map.data = self.map_msg.data
        self.width = self.map_msg.info.width
        self.height = self.map_msg.info.height
        self.resolution = self.map_msg.info.resolution

        self.new_map_as_np = StochOccupancyGrid2D(self.new_map.info.resolution, 
                                                       self.new_map.info.width, 
                                                      self.new_map.info.height, 
                                           self.new_map.info.origin.position.x, 
                                           self.new_map.info.origin.position.y, 
                                                                             5,                     
                                                             self.new_map.data)

        # Publish the map
        self.new_map_publisher = rospy.Publisher('/new_map', OccupancyGrid, queue_size=10)
        self.new_map_publisher.publish(self.new_map)

        # Prepare the combined map and the publisher
        self.combined_map = OccupancyGrid()
        self.combined_map.header = self.map_msg.header
        self.combined_map.info = self.map_msg.info
        self.combined_map_publisher = rospy.Publisher('/navigation_map', OccupancyGrid, queue_size=10)

        camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        camera_info = camera_info_msg.K
        camera_info = np.array(camera_info).reshape(3, 3)
        self.fx = camera_info[0, 0]
        self.fy = camera_info[1, 1]
        self.cx = camera_info[0, 2]
        self.cy = camera_info[1, 2]

        # Log-Probabilities to add or remove from the map 
        self.l_occ = 0.4
        self.l_free = np.log(0.35/0.65)

        self.is_turning = False

        self.proposed_objects = []
        self.detected_objects = []
        self.num_detected_objects = 0
        self.object_marker_array = MarkerArray()

        self.cone_map = np.ones((self.height, self.width))*-1
        self.new_cone_publisher = rospy.Publisher('/new_cone_map', DetectedObject, queue_size=10)

        self.object_publisher = rospy.Publisher('/object_array', MarkerArray, queue_size=10)
        

    def localized_callback(self, msg):
        self.is_localized = msg.data   

    def perceptual_field(self, x0, y0, x1, y1, all_x, all_y):
        """
        Determines the perceptual field of the camera in the global frame
        using Bresenham's line algorithm.

        x0/y0: global coordinates of the camera
        x1/y1: global coordinates of the end of the perceptual field (point cloud point)

        returns: list of tuples of the perceptual field
        """
        dx = np.abs(x1 - x0)
        if (x0 < x1):
            sx = 1
        else:
            sx = -1
        
        dy = -np.abs(y1 - y0)
        if (y0 < y1):
            sy = 1
        else:
            sy = -1

        perceptual_field = []
        e = dx + dy
        ignore_point = False
        while True:
            perceptual_field.append((x0, y0))
            if (x0 == x1 and y0 == y1):
                break

            if (x0, y0) in zip(all_x, all_y):
                ignore_point = True
                break
            
            e2 = 2 * e
            if e2 >= dy:
                if x0 == x1:
                    break
                e += dy
                x0 += sx
            
            if e2 <= dx:
                if y0 == y1:
                    break
                e += dx
                y0 += sy
        
        return perceptual_field, ignore_point

    def point_callback(self, msg):

        # If the robot is turning, don't do anything since the camera is not stable
        if self.is_turning:
            return

        # Get current position of the camera immediately so that it is as close to the point cloud as possible
        try:
            (translation, rotation) = self.trans_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
            camera_x = translation[0]
            camera_y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            camera_theta = euler[2]

            camera_location = (camera_x, camera_y, camera_theta)
            camera_offset = 0.127
            robot_location = (camera_x - camera_offset*np.cos(camera_theta), camera_y - camera_offset*np.sin(camera_theta), camera_theta)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("No location yet...", e)
            # Location not available yet
            return

        t1 = time.time()
        point_list = []
        height_list = []
        for point in reversed(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))):
            height = -point[1]+self.camera_height
            height_list.append(height)
            dist = np.sqrt(point[0]**2 + point[2]**2)

            if height <= self.total_height+0.5 and dist < 4:
                point_list.append([point[0], point[1], point[2]])
            else: # Points in general increase in height, so can stop adding as soon as we reach the total height
                break

        # x/y/z in camera frame (z=depth from camera)
        points_np = np.array(point_list)

        # If no points at all, return immediately
        if points_np.shape[0] == 0:
            return
        
        x_local = points_np[:, 0]
        y_local = points_np[:, 2]


        t2 = time.time()

        # Now map local coordinates to global coordinates
        theta_objects = np.arctan2(x_local, y_local)
        hypo = np.sqrt(x_local**2 + y_local**2)
        x_global = camera_location[0] + np.cos(camera_location[2] - theta_objects) * hypo
        y_global = camera_location[1] + np.sin(camera_location[2] - theta_objects) * hypo

        # Get coordinates in discrete map of x,y
        (x_global, y_global) = self.new_map_as_np.snap_to_grid1((x_global, y_global))

        # Get indices of the points in the map
        (x_global_indx, y_global_indx) = self.new_map_as_np.get_index((x_global, y_global))

        # Get unique points
        unique_points, unique_indx = np.unique(np.column_stack((x_global_indx, y_global_indx)), axis=0, return_index=True)
        theta_objects = theta_objects[unique_indx]
        hypo = hypo[unique_indx]

        # Get camera indices
        (camera_x, camera_y) = self.new_map_as_np.snap_to_grid((camera_x, camera_y))
        (camera_x_indx, camera_y_indx) = self.new_map_as_np.get_index((camera_x, camera_y))

        # Get perceptual field of the camera
        perceptual_field_indx = []
        indices_to_remove = []
        for i in range(unique_points.shape[0]):
            field, ignore = self.perceptual_field(camera_x_indx, camera_y_indx, unique_points[i, 0], unique_points[i, 1], unique_points[:, 0], unique_points[:, 1])
            
            # Since have multiple height levels, we need to remove points that are behind another point
            if ignore:
                indices_to_remove.append(i)
            else:
                perceptual_field_indx.extend(field)

        # Get footprint of the robot
        theta_array = np.arange(0, 2*np.pi, 0.1)
        robot_outline_x = []
        robot_outline_y = []
        for i in np.arange(0, self.robot_d/2.5, self.resolution):  # use 2.5 to be conservative
            robot_outline_x.append(i*np.cos(theta_array) + robot_location[0])
            robot_outline_y.append(i*np.sin(theta_array) + robot_location[1])
        robot_outline_x = np.array(robot_outline_x).flatten()
        robot_outline_y = np.array(robot_outline_y).flatten()
        (robot_outline_x, robot_outline_y) = self.new_map_as_np.snap_to_grid1((robot_outline_x, robot_outline_y))
        
        # Get indices of only the unique points as integers
        robot_outline = np.unique(np.column_stack((robot_outline_x, robot_outline_y)), axis=0) 
        robot_outline_x = robot_outline[:, 0]
        robot_outline_y = robot_outline[:, 1]
        (robot_outline_x_indx, robot_outline_y_indx) = self.new_map_as_np.get_index((robot_outline_x, robot_outline_y))
        robot_outline_x_indx = robot_outline_x_indx.astype(int)
        robot_outline_y_indx = robot_outline_y_indx.astype(int)        

        # Remove points that are behind another point
        unique_points = np.delete(unique_points, indices_to_remove, axis=0)
        theta_objects = np.delete(theta_objects, indices_to_remove)
        hypo = np.delete(hypo, indices_to_remove)

        # # Display the unique points
        # marker_array = MarkerArray()
        # i = 0
        # for j in range(unique_points.shape[0]):
        #     marker = Marker()
        #     marker.header.frame_id = "map"
        #     marker.type = Marker.SPHERE
        #     marker.action = Marker.ADD
        #     marker.id = i
        #     marker.scale.x = 0.1
        #     marker.scale.y = 0.1
        #     marker.scale.z = 0.1
        #     marker.color.a = 0.5
        #     marker.color.r = 0.0
        #     marker.color.g = 1.0
        #     marker.color.b = 0.0
        #     marker.pose.position.x = unique_points[j, 0] * self.resolution
        #     marker.pose.position.y = unique_points[j, 1] * self.resolution
        #     marker.pose.position.z = 0.1
        #     marker_array.markers.append(marker)
        #     i += 1

        r_objects = np.sqrt((unique_points[:, 0]* self.resolution - camera_location[0])**2 + (unique_points[:,1]* self.resolution - camera_location[1])**2)
        phi_objects = np.arctan2(unique_points[:,1]* self.resolution - camera_location[1], unique_points[:, 0]* self.resolution - camera_location[0]) - camera_location[2]

        # Only unique points in perceptual field
        perceptual_field_indx = np.unique(perceptual_field_indx, axis=0)
        perceptual_field_coords = np.array(perceptual_field_indx) * self.resolution

        # Display the perceptual field
        # marker_array = MarkerArray()
        # for j in range(perceptual_field_coords.shape[0]):
        #     marker = Marker()
        #     marker.header.frame_id = "map"
        #     marker.type = Marker.SPHERE
        #     marker.action = Marker.ADD
        #     marker.id = i
        #     marker.scale.x = 0.05
        #     marker.scale.y = 0.05
        #     marker.scale.z = 0.05
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.pose.position.x = perceptual_field_coords[j, 0]
        #     marker.pose.position.y = perceptual_field_coords[j, 1]
        #     marker.pose.position.z = 0.1
        #     marker_array.markers.append(marker)
        #     i += 1
        # marker_arr_pub.publish(marker_array)

        r = np.sqrt((perceptual_field_coords[:, 0] - camera_location[0])**2 + (perceptual_field_coords[:,1] - camera_location[1])**2)
        phi = np.arctan2(perceptual_field_coords[:,1] - camera_location[1], perceptual_field_coords[:, 0] - camera_location[0]) - camera_location[2]
        
        # Get the index of the object that is closest to the perceptual field
        k = np.argmin(np.abs(phi[:, np.newaxis] - phi_objects), axis=1)

        l = np.ones(len(k))
        
        # Free space
        indx = np.where(r <= r_objects[k])[0]
        l[indx] = self.l_free

        # Occupied space
        indx = np.where(np.logical_and(r_objects[k] <= 8, np.abs(r - r_objects[k]) < self.alpha/2))[0]
        l[indx] = self.l_occ

        # Ignore any really close points
        indx = np.where(r < 0.075)[0]
        l[indx] = self.l_free
        
        # Unknown space
        indx = np.where(np.logical_or(r > 8, r > r_objects[k] + self.alpha/2))[0]
        l[indx] = 0
        indx = np.where(np.abs(phi - phi_objects[k]) > self.beta/2)[0]
        l[indx] = 0
        
        t3 = time.time()

        self.new_map_as_np.l[perceptual_field_indx[:, 1].astype(int), perceptual_field_indx[:, 0].astype(int)] += l

        # Now set robot locations to lfree
        self.new_map_as_np.l[robot_outline_y_indx, robot_outline_x_indx] += 2*self.l_free

        self.new_map_as_np.recalculate_probs()
        self.new_map.data = (self.new_map_as_np.probs.flatten()*100).astype(int).tolist()
        self.new_map_publisher.publish(self.new_map)

        # combined_map_data = np.array(self.map_msg.data)
        map_data = np.array(self.map_msg.data)
        mod_data = np.array(self.map_mod_msg.data)
        combined_map_data = np.maximum(map_data, mod_data)
        combined_map_data = np.maximum(combined_map_data, self.cone_map.flatten())

        # new_map_binary = np.where(self.new_map_as_np.probs.flatten() > 0.85)[0]
        # combined_map_data[new_map_binary] = 100
        self.combined_map.data = combined_map_data.astype(int).tolist()
        self.combined_map_publisher.publish(self.combined_map) 

        
        # print("Intermediate Time taken: ", t2 - t1)
        # print("Total Time taken: ", t3 - t1)

    def cmd_vel_callback(self, msg):
        if np.abs(msg.angular.z) > 0.125:
            self.is_turning = True
        else:
            self.is_turning = False

    def detected_objects_callback(self, msg):
        object_array = msg.objects
        new_proposed_objects = []
        objs_to_pop = []

        for obj in object_array:
            already_exists = False
            x = obj.pose.position.x
            y = obj.pose.position.y

            for detected_obj in self.detected_objects:
                if np.sqrt((detected_obj[0] - x)**2 + (detected_obj[1] - y)**2) < obj.width:
                    already_exists = True
                    break

            if not already_exists:

                match_proposed = False
                # Check if we match with any proposed object
                for ii in range(len(self.proposed_objects)-1, -1, -1):
                    proposed_obj = self.proposed_objects[ii]
                    if np.sqrt((proposed_obj[0] - x)**2 + (proposed_obj[1] - y)**2) < obj.width/2:

                        self.proposed_objects[ii][3] += 1
                        self.proposed_objects[ii][5] = True
                        if self.proposed_objects[ii][3] > 5:

                            match_proposed = True
                            self.proposed_objects.pop(ii)

                            # Matches a proposed object, add to detected objects 
                            self.detected_objects.append([x, y, obj.width])
                            self.num_detected_objects += 1
                            print("Number of detected objects: ", self.num_detected_objects)

                            x_min = int((x - obj.width/2)/self.resolution)
                            x_max = int((x + obj.width/2)/self.resolution)
                            y_min = int((y - obj.width/2)/self.resolution)
                            y_max = int((y + obj.width/2)/self.resolution)
                            self.cone_map[y_min:y_max, x_min:x_max] = 100

                            marker = Marker()
                            marker.header.frame_id = "map"
                            marker.type = Marker.SPHERE
                            marker.action = Marker.ADD
                            marker.id = self.num_detected_objects
                            marker.scale.x = 0.1
                            marker.scale.y = 0.1
                            marker.scale.z = 0.1
                            marker.color.a = 1.0
                            marker.color.r = 0.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0
                            marker.pose.position.x = x
                            marker.pose.position.y = y
                            marker.pose.position.z = 0.1
                            self.object_marker_array.markers.append(marker)

                            cone_object = DetectedObject()
                            cone_object.class_name = "cone"
                            cone_object.pose.position.x = x
                            cone_object.pose.position.y = y
                            cone_object.pose.position.z = 0.0
                            cone_object.pose.orientation.w = 1
                            cone_object.width = obj.width
                            self.new_cone_publisher.publish(cone_object)

                            break
                
                if not match_proposed:
                    self.proposed_objects.append([x, y, obj.width, 0, 0, True])

        for ii in range(len(self.proposed_objects)-1, -1, -1):
            if self.proposed_objects[ii][5] == False:
                self.proposed_objects[ii][4] += 1
                if self.proposed_objects[ii][4] > 10:
                    self.proposed_objects.pop(ii)
            else:
                self.proposed_objects[ii][5] = False

        self.object_publisher.publish(self.object_marker_array)

    def object_from_agent_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        width = msg.width
        self.detected_objects.append([x, y, width])
        self.num_detected_objects += 1
        print("Added object from other agent")
        print("Number of detected objects: ", self.num_detected_objects)

        x_min = int((x - width/2)/self.resolution)
        x_max = int((x + width/2)/self.resolution)
        y_min = int((y - width/2)/self.resolution)
        y_max = int((y + width/2)/self.resolution)
        self.cone_map[y_min:y_max, x_min:x_max] = 100

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = self.num_detected_objects
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        self.object_marker_array.markers.append(marker)     
        self.object_publisher.publish(self.object_marker_array)

    def run(self):
        while not self.is_localized:
            rospy.sleep(1)
        self.cmd_vel_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        self.point_cloud_subscriber = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.point_callback, queue_size=1)
        self.detected_object_subscriber = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=10)
        self.object_from_agent_subscriber = rospy.Subscriber('/object_from_agent', DetectedObject, self.object_from_agent_callback, queue_size=10)
        rospy.spin()

    def shutdown(self):
        rospy.loginfo("Shutting down Occupancy Grid Mapper")
        

if __name__ == '__main__':
    rospy.init_node('occupancy_grid_mapper', anonymous=True)
    rospy.loginfo("Occupancy Grid Mapper Started")

    marker_arr_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

    my_map = Map()
    rospy.on_shutdown(my_map.shutdown)
    my_map.run()
