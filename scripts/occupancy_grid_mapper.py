import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import tf
import sensor_msgs.point_cloud2 as pc2
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, Person, PersonArray

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

        self.person_static_map = np.ones((self.height, self.width))*-1  # Map for tracking people
        self.person_moving_map = np.ones((self.height, self.width))*-1  # Map for tracking people
        self.person_dict = dict()  # Dictionary for tracking people
        self.person_subscriber = rospy.Subscriber('/person', PersonArray, self.person_callback, queue_size=10) 

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

        # Get data from msg into an numpy array
        data = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        dist = np.sqrt(data[:, 0]**2 + data[:, 2]**2)
        height = -data[:, 1] + self.camera_height        
        
        # Get only the points that are within the height and distance range
        indices = np.where(np.logical_and(height <= self.total_height+0.5, dist < 4))[0]
        points_np = data[indices, :]
        
        # point_list = []
        # height_list = []
        # for point in reversed(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))):
        #     height = -point[1]+self.camera_height
        #     height_list.append(height)
        #     dist = np.sqrt(point[0]**2 + point[2]**2)

        #     if height <= self.total_height+0.5 and dist < 4:
        #         point_list.append([point[0], point[1], point[2]])
        #     else: # Points in general increase in height, so can stop adding as soon as we reach the total height
        #         break

        # # x/y/z in camera frame (z=depth from camera)
        # points_np = np.array(point_list)

        # If no points at all, return immediately
        if points_np.shape[0] == 0:
            return
        
        x_local = points_np[:, 0]
        y_local = points_np[:, 2]


        t2 = time.time()

        # print("Time taken to get points: ", t2 - t1)

        # Now map local coordinates to global coordinates
        theta_objects = np.arctan2(x_local, y_local)
        hypo = np.sqrt(x_local**2 + y_local**2)

        sorted_indx = np.argsort(theta_objects)
        min_angle = theta_objects[sorted_indx[0]]
        max_angle = theta_objects[sorted_indx[-1]]

        angle_indx = 0
        angle = min_angle
        max_dist = hypo[sorted_indx[0]]

        x_global = []
        y_global = []

        for i in range(1, len(sorted_indx)):
            
            angle_diff = theta_objects[sorted_indx[i]] - angle
            if angle_diff > 0.01:
                x_global.append(camera_location[0] + np.cos(camera_location[2] - theta_objects[sorted_indx[angle_indx]]) * max_dist)
                y_global.append(camera_location[1] + np.sin(camera_location[2] - theta_objects[sorted_indx[angle_indx]]) * max_dist)

                if i < len(sorted_indx)-1:
                    angle = theta_objects[sorted_indx[i]]
                    max_dist = hypo[sorted_indx[i]]
                    angle_indx = i
            else:
                if hypo[sorted_indx[i]] > max_dist:
                    max_dist = hypo[sorted_indx[i]]
                    angle_indx = i

        x_global = np.array(x_global)
        y_global = np.array(y_global)
        # print("Number of points: ", x_global.shape[0])

        # x_global = camera_location[0] + np.cos(camera_location[2] - theta_objects) * hypo
        # y_global = camera_location[1] + np.sin(camera_location[2] - theta_objects) * hypo
        # print("Full Number of points: ", x_global.shape[0])

        # Get coordinates in discrete map of x,y
        (x_global, y_global) = self.new_map_as_np.snap_to_grid1((x_global, y_global))

        # Get indices of the points in the map
        (x_global_indx, y_global_indx) = self.new_map_as_np.get_index((x_global, y_global))

        # Get unique points
        unique_points, unique_indx = np.unique(np.column_stack((x_global_indx, y_global_indx)), axis=0, return_index=True)

        # Get camera indices
        (camera_x, camera_y) = self.new_map_as_np.snap_to_grid((camera_x, camera_y))
        (camera_x_indx, camera_y_indx) = self.new_map_as_np.get_index((camera_x, camera_y))

        t21 = time.time()
        # print("Preprocessing TIme: ", t21 - t2)

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

        t22 = time.time()
        # print("Perceptual Field Time: ", t22 - t21)

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

        t23 = time.time()
        # print("Robot Outline Time: ", t23 - t22)

        # Remove points that are behind another point
        unique_points = np.delete(unique_points, indices_to_remove, axis=0)

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
        
        # Remove indx that are outside the map
        indx = np.where(np.logical_or(perceptual_field_indx[:, 0] >= self.width, perceptual_field_indx[:, 1] >= self.height))[0]
        perceptual_field_indx = np.delete(perceptual_field_indx, indx, axis=0)

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
        # print("final processing time: ", t3 - t23)

        self.new_map_as_np.l[perceptual_field_indx[:, 1].astype(int), perceptual_field_indx[:, 0].astype(int)] += l 

        # Now set robot locations to lfree
        self.new_map_as_np.l[robot_outline_y_indx, robot_outline_x_indx] += 2*self.l_free

        self.new_map_as_np.recalculate_probs()
        self.new_map.data = (self.new_map_as_np.probs.flatten()*100).astype(int).tolist()
        self.new_map_publisher.publish(self.new_map)

        # combined_map_data = np.array(self.map_msg.data)
        map_data = np.array(self.map_msg.data).reshape(self.height, self.width)
        mod_data = np.array(self.map_mod_msg.data).reshape(self.height, self.width)
        combined_map_data = np.maximum(map_data, mod_data)
        combined_map_data = np.maximum(combined_map_data, self.cone_map)
        combined_map_data = np.maximum(combined_map_data, self.person_static_map)

        # We only care about the person_moving_map if within 2 meters of us:
        x_min = max([int((camera_location[0] - 1)/self.resolution), 0])
        x_max = min([int((camera_location[0] + 1)/self.resolution), self.width])
        y_min = max([int((camera_location[1] - 1)/self.resolution), 0])
        y_max = min([int((camera_location[1] + 1)/self.resolution), self.height])
        combined_map_data[y_min:y_max, x_min:x_max] = np.maximum(combined_map_data[y_min:y_max, x_min:x_max], self.person_moving_map[y_min:y_max, x_min:x_max])

        # new_map_binary = np.where(self.new_map_as_np.probs.flatten() > 0.85)[0]
        # combined_map_data[new_map_binary] = 100
        self.combined_map.data = combined_map_data.flatten().astype(int).tolist()
        self.combined_map_publisher.publish(self.combined_map) 

        
        # print("Intermediate Time taken: ", t2 - t1)
        # print("Processing Time taken: ", t3 - t2)
        # print("Total Time taken: ", t3 - t1)
        # print(" ")

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

            # Temporary while we figure out what we want to do
            if obj.class_name != "cone":
                continue

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
                            print("Number of detected objects: ", self.num_detected_objects - self.num_removed_objects)

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

        for detected_obj in self.detected_objects:
            if np.sqrt((detected_obj[0] - x)**2 + (detected_obj[1] - y)**2) < width:
                return

        self.detected_objects.append([x, y, width])
        self.num_detected_objects += 1
        print("Added object from other agent")
        print("Number of detected objects: ", self.num_detected_objects - self.num_removed_objects)

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

    def object_from_sensor_callback(self, msg):
        object_array = msg.objects
        sensor_id = msg.sending_agent

        sensed_object_indx = []

        new_object_list = []

        for obj in object_array:
            x = obj.pose.position.x
            y = obj.pose.position.y
            width = obj.width

            new_object_list.append([x, y, width])

            obj_exists = False
            for detected_obj in self.detected_objects:
                if np.sqrt((detected_obj[0] - x)**2 + (detected_obj[1] - y)**2) < width:
                    obj_exists = True

            obj_num = 0
            if sensor_id in list(self.sensor_objects.keys()):
                for sens_obj in self.sensor_objects[sensor_id]:
                    if np.sqrt((sens_obj[0] - x)**2 + (sens_obj[1] - y)**2) < width:
                        sensed_object_indx.append(obj_num)
                        break
                    obj_num += 1

            if not obj_exists:
                self.detected_objects.append([x, y, width])
                self.num_detected_objects += 1
                print("Added object from sensor")
                print("Number of detected objects: ", self.num_detected_objects - self.num_removed_objects)

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

        # Remove the objects that were not sensed
        if sensor_id in list(self.sensor_objects.keys()):
            for i in range(len(self.sensor_objects[sensor_id])):
                if i not in sensed_object_indx:
                    old_x, old_y, old_width = self.sensor_objects[sensor_id][i]
                    for j in range(len(self.detected_objects)):
                        obj  = self.detected_objects[j]
                        if np.sqrt((obj[0] - old_x)**2 + (obj[1] - old_y)**2) < old_width:
                            self.detected_objects.remove(obj)
                            x_min = int((obj[0] - obj[2]/2)/self.resolution)
                            x_max = int((obj[0] + obj[2]/2)/self.resolution)
                            y_min = int((obj[1] - obj[2]/2)/self.resolution)
                            y_max = int((obj[1] + obj[2]/2)/self.resolution)
                            self.cone_map[y_min:y_max, x_min:x_max] = 0
                            break
                    # Remove the marker
                    print("Removed object from sensor")
                    self.num_removed_objects += 1

                    # Remove corresponding marker
                    for j in range(len(self.object_marker_array.markers)):
                        marker = self.object_marker_array.markers[j]
                        if np.sqrt((marker.pose.position.x - old_x)**2 + (marker.pose.position.y - old_y)**2) < old_width:
                            self.object_marker_array.markers[j].action = Marker.DELETE
                            break
                    self.object_publisher.publish(self.object_marker_array)

        # Update the sensor objects
        self.sensor_objects[sensor_id] = new_object_list

    def person_callback(self, msg):        
        person_array = msg.persons
        for person in person_array:
            person_id = person.id
            x = person.pose.position.x
            y = person.pose.position.y
            width = person.width
            is_static = person.static
            is_exited = person.exited

            width = max([min([width, 0.15]), 0.45])

            x_min = max([int((x - width/2)/self.resolution), 0])
            x_max = min([int((x + width/2)/self.resolution), self.width])
            y_min = max([int((y - width/2)/self.resolution), 0])
            y_max = min([int((y + width/2)/self.resolution), self.height])

            # Get the previous location
            already_exists = False
            if person_id in list(self.person_dict.keys()):
                x_min_p, x_max_p, y_min_p, y_max_p, is_static_p = self.person_dict[person_id]
                already_exists = True


            if is_exited:
                if is_static_p:
                    self.person_static_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0
                else:
                    self.person_moving_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0

                self.person_dict.pop(person_id)
            elif already_exists:
                if is_static:
                        
                    # Clear Previous maps
                    if is_static_p:
                        self.person_static_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0
                    else:
                        self.person_moving_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0

                    # Update new maps
                    self.person_static_map[y_min:y_max, x_min:x_max] = 100
                else: # Not static

                    # Clear Previous maps
                    if is_static_p:
                        self.person_static_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0
                    else:
                        self.person_moving_map[y_min_p:y_max_p, x_min_p:x_max_p] = 0
                    
                    # Update new maps
                    self.person_moving_map[y_min:y_max, x_min:x_max] = 100

                # Update the dictionary
                self.person_dict[person_id] = [x_min, x_max, y_min, y_max, is_static]
            else:
                if is_static:
                    self.person_static_map[y_min:y_max, x_min:x_max] = 100
                else:
                    self.person_moving_map[y_min:y_max, x_min:x_max] = 100
                self.person_dict[person_id] = [x_min, x_max, y_min, y_max, is_static]

    def run(self):
        while not self.is_localized:
            rospy.sleep(1)

        self.sensor_objects = dict()
        self.num_removed_objects = 0

        self.cmd_vel_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        self.point_cloud_subscriber = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.point_callback, queue_size=1)
        self.detected_object_subscriber = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=10)
        self.object_from_agent_subscriber = rospy.Subscriber('/object_from_agent', DetectedObject, self.object_from_agent_callback, queue_size=10)
        self.object_from_sensor_subscriber = rospy.Subscriber('/object_from_sensor', DetectedObjectArray, self.object_from_sensor_callback, queue_size=10)
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
