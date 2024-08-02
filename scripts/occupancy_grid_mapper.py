import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import tf
import sensor_msgs.point_cloud2 as pc2

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

    def __init__(self, camera_height=0.216, total_height=0.3175):

        self.camera_height = camera_height
        self.total_height = total_height

        # Tuning parameters for the inverse range sensor model
        self.alpha = 0.2   # 0.1 meters
        self.beta = 0.035  # 2 radians

        self.trans_listener = tf.TransformListener()

        # Get current map created from LIDAR SLAM
        self.map_msg = rospy.wait_for_message("/map", OccupancyGrid)
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
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

    def rgbd_callback(self, depth_data):
        rospy.loginfo("Received rgbd data")
        
        # Get current position of the camera 
        try:
            (translation, rotation) = self.trans_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
            camera_x = translation[0]
            camera_y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            camera_theta = euler[2]

            camera_location = (camera_x, camera_y, camera_theta)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("No location yet...", e)
            # Location not available yet
            return

        # Get the rgbd data
        depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)
        # Get x/y/z coordinates from depth data
        x = np.arange(0, depth_data.width)
        y = np.arange(0, depth_data.height)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = depth.flatten()
        x = (x - self.cx) * z / self.fx / 1000
        y = (y - self.cy) * z / self.fy / 1000
        z = z / 1000

        # Now map x/y/z to global coordinates using camera location
        theta_objects = np.arctan2(x, z)
        hypo = np.sqrt(x**2 + z**2)
        x_global = camera_location[0] + np.cos(camera_location[2] - theta_objects) * hypo
        y_global = camera_location[1] + np.sin(camera_location[2] - theta_objects) * hypo
        z_global = y + self.camera_height

        plt.scatter(x_global, y_global)
        plt.savefig('scatter.png')

        # Only use valid z measurements!
        indx = np.where(z > 0)
        x_global = x_global[indx]
        y_global = y_global[indx]
        z_global = z_global[indx]

        indx = np.where(z_global/1000 <= self.total_height)
        x_global = x_global[indx]
        y_global = y_global[indx]
        z_global = z_global[indx]

        x_global = x_global / 1000  # meters
        y_global = y_global / 1000  # meters
        z_global = z_global / 1000  # meters

        # Get coordinates in discrete map of x,y
        (x_global, y_global) = self.new_map_as_np.snap_to_grid1((x_global, y_global))

        

        # Get indices of the points in the map
        (x_global_indx, y_global_indx) = self.new_map_as_np.get_index((x_global, y_global))

        # Get unique points
        unique_points = np.unique(np.column_stack((x_global_indx, y_global_indx)), axis=0)

        # Get camera indices
        (camera_x, camera_y) = self.new_map_as_np.snap_to_grid((camera_x, camera_y))
        (camera_x_indx, camera_y_indx) = self.new_map_as_np.get_index((camera_x, camera_y))

        

        # Get perceptual field of the camera
        perceptual_field = []
        for i in range(unique_points.shape[0]):
            field = self.perceptual_field(camera_x_indx, camera_y_indx, unique_points[i, 0], unique_points[i, 1])
            perceptual_field.extend(field)

        

        print(len(perceptual_field))

        # TODO: Only unique points in perceptual field
        perceptual_field = np.unique(perceptual_field, axis=0)

        print((perceptual_field))

        # self.occupancy_grid_mapper(camera_location, x_global, y_global, z_global)
    
    def occupancy_grid_mapper(self, camera_location, x_measurement, y_measurement, z_measurement):

        # Calculate angle between camera and measurement points (in global frame)
        thetas = np.arctan2(y_measurement - camera_location[1], x_measurement - camera_location[0])
        thetas = np.arctan2(np.sin(thetas), np.cos(thetas))

        # calculate distance between camera and measurement points
        distances = np.sqrt((x_measurement - camera_location[0])**2 + (y_measurement - camera_location[1])**2)

        # Meshgrid for the map
        i = np.arange(0, self.width) * self.resolution
        j = np.arange(0, self.height) * self.resolution
        i, j = np.meshgrid(i, j)

        x, y = self.new_map_as_np.snap_to_grid1((i, j))

        # Implementation of inverse_range_sensor_model from Probabilistic Robotics, Table 9.2
        r = np.sqrt((x - camera_location[0])**2 + (y - camera_location[1])**2)
        phi = np.arctan2(y - camera_location[1], x - camera_location[0]) - camera_location[2]
        phi = np.arctan2(np.sin(phi), np.cos(phi))


        print(np.min(phi), np.max(phi))

        if np.max(thetas) > np.pi - 0.1 and np.min(thetas) < -np.pi + 0.1:
            pass
            # TODO
        else:
            max_theta = np.max(thetas)
            min_theta = np.min(thetas)

            indx = np.where(np.logical_and(phi >= min_theta, phi <= max_theta, r < 5))
            phi = phi[indx]
            r = r[indx]

            k = np.argmin(np.abs(phi[:, np.newaxis] - thetas), axis=1)
            print(len(k))
            print(len(phi))

        # print(np.min(phi), np.max(phi))

        # k = np.argmin(np.abs(thetas - phi[i]))

        # for i in range(self.width):
        #     for j in range(self.height):
        #         x, y = self.new_map_as_np.snap_to_grid1((i, j))

        #         # Implementation of inverse_range_sensor_model from Probabilistic Robotics, Table 9.2
        #         r = np.sqrt((x - camera_location[0])**2 + (y - camera_location[1])**2)
        #         phi = np.arctan2(y - camera_location[1], x - camera_location[0]) - camera_location[2]

        #         k = np.argmin(np.abs(thetas - phi))

        #         if r > np.min([distances[k] + self.alpha/2, 5]):
        #             continue
        #         elif np.abs(phi - thetas[k]) > self.beta/2:
        #             continue
        #         elif distances[k] < 5 and np.abs(r - distances[k]) < self.alpha/2:
        #             self.new_map_as_np.l[j, i] += self.l_occ
        #         elif r <= distances[k]:
        #             self.new_map_as_np.probs[j, i] += self.l_free

        self.new_map_as_np.recalculate_probs()
        self.new_map.data = self.new_map_as_np.probs.flatten()
        self.new_map_publisher.publish(self.new_map)                

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

        print("Received Message")

        # Get current position of the camera immediately so that it is as close to the point cloud as possible
        try:
            (translation, rotation) = self.trans_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
            camera_x = translation[0]
            camera_y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            camera_theta = euler[2]

            camera_location = (camera_x, camera_y, camera_theta)

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

            # Points in general increase in height, so can stop adding as soon as we reach the total height
            if height <= self.total_height and dist < 3:
                point_list.append([point[0], point[1], point[2]])
            else:
                break

        # x/y/z in camera frame (z=depth from camera)
        points_np = np.array(point_list)

        # # Only use points within a certain height
        # indx = np.where(points_np[:, 1] <= self.total_height)
        # points_np = points_np[indx]

        # Only use points within a certain distance
        # dist = np.sqrt(points_np[:, 0]**2 + points_np[:, 2]**2)
        # indx = np.where(dist < 3)
        # points_np = points_np[indx]

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

        # Remove points that are behind another point
        unique_points = np.delete(unique_points, indices_to_remove, axis=0)
        theta_objects = np.delete(theta_objects, indices_to_remove)
        hypo = np.delete(hypo, indices_to_remove)

        # Display the unique points
        marker_array = MarkerArray()
        i = 0
        for j in range(unique_points.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = unique_points[j, 0] * self.resolution
            marker.pose.position.y = unique_points[j, 1] * self.resolution
            marker.pose.position.z = 0.1
            marker_array.markers.append(marker)
            i += 1
        # marker_arr_pub.publish(marker_array)

        r_objects = np.sqrt((unique_points[:, 0]* self.resolution - camera_location[0])**2 + (unique_points[:,1]* self.resolution - camera_location[1])**2)
        phi_objects = np.arctan2(unique_points[:,1]* self.resolution - camera_location[1], unique_points[:, 0]* self.resolution - camera_location[0]) - camera_location[2]

        # Only unique points in perceptual field
        perceptual_field_indx = np.unique(perceptual_field_indx, axis=0)
        perceptual_field_coords = np.array(perceptual_field_indx) * self.resolution

        # Display the perceptual field
        # marker_array = MarkerArray()
        for j in range(perceptual_field_coords.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = perceptual_field_coords[j, 0]
            marker.pose.position.y = perceptual_field_coords[j, 1]
            marker.pose.position.z = 0.1
            marker_array.markers.append(marker)
            i += 1
        marker_arr_pub.publish(marker_array)

        r = np.sqrt((perceptual_field_coords[:, 0] - camera_location[0])**2 + (perceptual_field_coords[:,1] - camera_location[1])**2)
        phi = np.arctan2(perceptual_field_coords[:,1] - camera_location[1], perceptual_field_coords[:, 0] - camera_location[0]) - camera_location[2]
        # k = np.argmin(np.abs(phi[:, np.newaxis] - theta_objects), axis=1)
        k = np.argmin(np.abs(phi[:, np.newaxis] - phi_objects), axis=1)
        print(phi.shape)
        print(phi_objects.shape)
        print(phi_objects)

        l = np.ones(len(k))
        

        indx = np.where(r <= r_objects[k])[0]
        l[indx] = self.l_free
        print(len(indx))

        indx = np.where(np.logical_and(r_objects[k] <= 8, np.abs(r - r_objects[k]) < self.alpha/2))[0]
        l[indx] = self.l_occ
        print(len(indx))
        
        indx = np.where(np.logical_or(r > 8, r > r_objects[k] + self.alpha/2))[0]
        l[indx] = 0
        print(len(indx))
        indx = np.where(np.abs(phi - phi_objects[k]) > self.beta/2)[0]
        l[indx] = 0
        print(len(indx))
        

        t3 = time.time()

        # TODO: Make sure this update is actually correct...
        self.new_map_as_np.l[perceptual_field_indx[:, 1].astype(int), perceptual_field_indx[:, 0].astype(int)] += l
        self.new_map_as_np.recalculate_probs()
        self.new_map.data = (self.new_map_as_np.probs.flatten()*100).astype(int).tolist()
        self.new_map_publisher.publish(self.new_map)

        combined_map_data = np.array(self.map_msg.data)
        new_map_binary = np.where(self.new_map_as_np.probs.flatten() > 0.5)[0]
        combined_map_data[new_map_binary] = 100
        self.combined_map.data = combined_map_data.astype(int).tolist()
        self.combined_map_publisher.publish(self.combined_map) 
        
        print("Intermediate Time taken: ", t2 - t1)
        print("Total Time taken: ", t3 - t1)

    def run(self):

        # self.rgbd_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.rgbd_callback)
        self.point_cloud_subscriber = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.point_callback, queue_size=1)
        rospy.spin()
        

if __name__ == '__main__':
    rospy.init_node('occupancy_grid_mapper', anonymous=True)
    rospy.loginfo("Occupancy Grid Mapper Started")

    marker_arr_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

    my_map = Map()
    my_map.run()
