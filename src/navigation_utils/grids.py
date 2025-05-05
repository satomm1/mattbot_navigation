import numpy as np

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
                window_size, probs, thresh=0.5, robot_d=0.6):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.window_size = 10 # window_size
        # print(window_size)
        self.thresh = thresh
        self.robot_d=robot_d

    def __add__(self, other):
        if not isinstance(other, StochOccupancyGrid2D):
            raise TypeError("Can only add another StochOccupancyGrid2D instance")
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must have the same dimensions to be added")
        
        new_probs = np.maximum(self.probs, other.probs)
        return StochOccupancyGrid2D(self.resolution, self.width, self.height,
                                    self.origin_x, self.origin_y,
                                    self.window_size, new_probs, self.thresh, self.robot_d)

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

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