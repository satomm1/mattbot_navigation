import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
import tf

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

    def recalculate_probs(self):
        self.probs = 1 - 1/(1+np.exp(self.l))

class Map:

    def __init__(self, camera_height=0.216, total_height=0.3175):

        self.camera_height = camera_height
        self.total_height = total_height

        # Tuning parameters for the inverse range sensor model
        self.alpha = 0.1   # 0.1 meters
        self.beta = 0.035  # 2 radians

        self.trans_listener = tf.TransformListener()

        # Get current map created from LIDAR SLAM
        map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        self.new_map = OccupancyGrid()
        self.new_map.header = map_msg.header
        self.new_map.info = map_msg.info
        self.new_map.data = map_msg.data
        self.width = map_msg.info.width
        self.height = map_msg.info.height
        self.resolution = map_msg.info.resolution

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
        x = (x - self.cx) * z / self.fx
        y = (y - self.cy) * z / self.fy
        
        # Now map x/y/z to global coordinates using camera location
        theta_objects = np.arctan2(x, z)
        hypo = np.sqrt(x**2 + z**2)
        x_global = camera_location[0] + np.cos(camera_location[2] - theta_objects) * hypo
        y_global = camera_location[1] + np.sin(camera_location[2] - theta_objects) * hypo
        z_global = y + self.camera_height

        # Only use valid z measurements!
        indx = np.where(z > 0)
        x_global = x_global[indx]
        y_global = y_global[indx]
        z_global = z_global[indx]

        indx = np.where(z_global <= self.total_height)
        x_global = x_global[indx]
        y_global = y_global[indx]
        z_global = z_global[indx]

        x_global = x_global / 1000  # meters
        y_global = y_global / 1000  # meters
        z_global = z_global / 1000  # meters

        self.occupancy_grid_mapper(camera_location, x_global, y_global, z_global)
    
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

    def run(self):

        self.rgbd_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.rgbd_callback)
        rospy.spin()
        

if __name__ == '__main__':
    rospy.init_node('occupancy_grid_mapper', anonymous=True)
    rospy.loginfo("Occupancy Grid Mapper Started")

    my_map = Map()
    my_map.run()
