import numpy as np

from social_path_planning import (
    attach_wall_distance_cache,
    DEFAULT_DIST_THRESH,
    travel_dir_to_dir_idx
)

# A 2D state space grid with a set of rectangular obstacles. The grid is fully deterministic
class DetOccupancyGrid2D(object):
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

class StochOccupancyGrid2D(object):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                window_size, probs, thresh=0.5, robot_d=0.6, wall_distance_cache_path=None,
                wall_distance_cache_use_ros_layout=True):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.l = np.zeros((height, width))
        self.window_size = 10 # window_size
        # print(window_size)
        self.thresh = thresh
        self.robot_d=robot_d
        self._d_right = None  # Cache for distance to the nearest wall on the right
        self._wall_distance_cache_use_ros_layout = wall_distance_cache_use_ros_layout

        self.extent = [self.origin_x, self.origin_x + self.width * self.resolution,
                       self.origin_y, self.origin_y + self.height * self.resolution]

        attach_wall_distance_cache(
            self,
            wall_distance_cache_path,
            auto_build=False,
            use_ros_cache_layout=wall_distance_cache_use_ros_layout,
        )

    def __add__(self, other):
        if not isinstance(other, StochOccupancyGrid2D):
            raise TypeError("Can only add another StochOccupancyGrid2D instance")
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must have the same dimensions to be added")
        
        new_probs = np.maximum(self.probs, other.probs)
        return StochOccupancyGrid2D(
            self.resolution,
            self.width,
            self.height,
            self.origin_x,
            self.origin_y,
            self.window_size,
            new_probs,
            self.thresh,
            self.robot_d,
            wall_distance_cache_path=None,
            wall_distance_cache_use_ros_layout=self._wall_distance_cache_use_ros_layout,
        )

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def snap_to_grid1(self, x):
        return (self.resolution * np.round(x[0] / self.resolution), self.resolution * np.round(x[1] / self.resolution))

    def get_index(self, x):
        return (np.round((x[0]-self.origin_x)/self.resolution), np.round((x[1]-self.origin_y)/self.resolution))

    def recalculate_probs(self):
        self.probs = 1 - 1/(1+np.exp(self.l))

    def decay_l(self):
        indx = np.where(np.abs(self.l) < 2)
        self.l[indx] = 0.85*self.l[indx]

    def set_occupied(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Create a mask for indices where l > 1
        mask = self.l[y, x] >= 1
        
        # Update values where the mask is True
        self.l[y[mask], x[mask]] += 3
        # Ensure we never go more than 10
        self.l[y[mask], x[mask]] = np.minimum(self.l[y[mask], x[mask]], 10)
        
        # Update values where the mask is False
        self.l[y[~mask], x[~mask]] = 1.2

    def is_free(self, state):
        # combine the probabilities of each cell by assuming independence
        # of each estimation
        x, y = self.snap_to_grid(state)
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        
        # Check if in configuration space
        half_config_size = int(round(self.robot_d/2/self.resolution))
        #print(half_config_size)
        config_x_lower = max(0, grid_x - half_config_size)
        config_y_lower = max(0, grid_y - half_config_size)
        config_x_upper = min(self.width, grid_x + half_config_size)
        config_y_upper = min(self.height, grid_y + half_config_size)
        if np.sum(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]>85):
            # values, counts = np.unique(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper], return_counts=True)
            # print(values)
            # print(counts)
            #print(self.resolution)
            return False  # Not free according to configuration space
        # elif np.sum(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]<0):
        #     return False  # This is unknown space, not free!

        # Now check probabilities
        half_size = int(round((self.window_size-1)/2))
        grid_x_lower = max(0, grid_x - half_size)
        grid_y_lower = max(0, grid_y - half_size)
        grid_x_upper = min(self.width, grid_x + half_size + 1)
        grid_y_upper = min(self.height, grid_y + half_size + 1)        
        
        prob_window = self.probs[grid_y_lower:grid_y_upper, grid_x_lower:grid_x_upper]
        p_total = np.prod(1. - np.maximum(prob_window / 100., 0.))

        return (1. - p_total) < self.thresh
    
    def update(self, new_probs):
        if new_probs.shape != self.probs.shape:
            raise ValueError("New probabilities must have the same shape as the existing grid")
        self.probs = new_probs

    def get_probs(self):
        return self.probs
    
    def dist_to_wall_left(self, x, travel_dir, dist_thresh=15.0):
        return self.dist_to_wall_right(x, [-travel_dir[0], -travel_dir[1]], dist_thresh=dist_thresh)

    def dist_to_wall_right(self, x, travel_dir, dist_thresh=15.0):
        """
        Return distance (meters) from world position x=(x,y) to the first occupied or
        unknown cell found to the right of the travel_dir. If no wall is found inside
        the map bounds, returns 0.
        """
        if (
            self._d_right is not None
            and abs(float(dist_thresh) - DEFAULT_DIST_THRESH) < 1e-9
        ):
            col = int(np.round((x[0] - self.origin_x) / self.resolution))
            row = int(np.round((x[1] - self.origin_y) / self.resolution))
            col = int(np.clip(col, 0, self.width - 1))
            row = int(np.clip(row, 0, self.height - 1))
            k = travel_dir_to_dir_idx(travel_dir)
            return float(self._d_right[row, col, k])

        return self._dist_to_wall_right_raycast(x, travel_dir, dist_thresh=dist_thresh)

    def _dist_to_wall_right_raycast(self, x, travel_dir, dist_thresh=15.0):
        """Ray-march implementation used for planning when no cache and for precompute."""
        right = np.array([travel_dir[1], -travel_dir[0]], dtype=float)
        rn = np.linalg.norm(right)
        if rn < 1e-12:
            return 100.0
        right = right / rn

        # map bounds in meters
        x_min = self.origin_x
        x_max = self.origin_x + self.width * self.resolution
        y_min = self.origin_y
        y_max = self.origin_y + self.height * self.resolution

        # sampling parameters
        step = max(self.resolution * 0.25, 1e-4)   # quarter cell steps (or small eps)
        # maximum distance to search: distance to map border along right direction
        # compute intersection of ray x + t*right with map bounding box to get a safe upper bound
        t_max = 0.0
        # compute candidate distances to each vertical boundary
        if right[0] > 0:
            t_max = max(t_max, (x_max - x[0]) / right[0])
        elif right[0] < 0:
            t_max = max(t_max, (x_min - x[0]) / right[0])
        if right[1] > 0:
            t_max = max(t_max, (y_max - x[1]) / right[1])
        elif right[1] < 0:
            t_max = max(t_max, (y_min - x[1]) / right[1])
        # if right component is 0 that boundary doesn't constrain; t_max may remain 0 if both comps lead outward.
        # ensure positive t_max
        if t_max <= 0:
            # ray immediately points out of bounds — no wall reachable to the right inside the map
            return 100
        if t_max > dist_thresh:
            t_max = dist_thresh

        # sample along ray
        n_steps = int(np.ceil(t_max / step))
        px = float(x[0])
        py = float(x[1])
        for i in range(1, n_steps + 1):
            t = i * step
            sx = px + right[0] * t
            sy = py + right[1] * t
            # check bounds (small numerical tolerance)
            if sx < x_min or sx >= x_max or sy < y_min or sy >= y_max:
                return 100
            # convert to grid indices (row, col). self.probs indexed as [row(y), col(x)]
            col = int(np.floor((sx - self.origin_x) / self.resolution))
            row = int(np.floor((sy - self.origin_y) / self.resolution))
            # clamp safety
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                return 100
            p = self.probs[row, col]
            if p >= self.thresh or p < 0:
                return t
        return 100
