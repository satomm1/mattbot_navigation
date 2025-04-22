import numpy as np

V_PREV_THRES = 0.0001

def wrapToPi(a):
    if isinstance(a, list):
        return [(x+np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """

    def __init__(self, kpx, kpy, kdx, kdy,
                 V_max=0.6, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8)  # Polynomial coefficients for x(t) and y(t) as
        # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t, self.traj_times, self.traj[:, 0])
        y_d = np.interp(t, self.traj_times, self.traj[:, 1])
        xd_d = np.interp(t, self.traj_times, self.traj[:, 3])
        yd_d = np.interp(t, self.traj_times, self.traj[:, 4])
        xdd_d = np.interp(t, self.traj_times, self.traj[:, 5])
        ydd_d = np.interp(t, self.traj_times, self.traj[:, 6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        if self.V_prev < V_PREV_THRES:
            self.V_prev = np.sqrt(xd_d ** 2 + yd_d ** 2)

        x_dot = self.V_prev * np.cos(th)
        y_dot = self.V_prev * np.sin(th)

        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

        a = u1 * np.cos(th) + u2 * np.sin(th)
        om = -u1 * np.sin(th) / self.V_prev + u2 * np.cos(th) / self.V_prev

        V = self.V_prev + a * dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # If near the end of the trajectory, slow down so we don't stop abruptly
        x_goal, _, _, y_goal, _, _ = self.get_desired_state(self.traj_times[-1])
        dist = np.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)
        if dist < 1:
            new_V_max = 0.25+0.25*dist # Slow down linearly starting at 0.5V_max to 0.25V_max
            V = np.clip(V, -new_V_max, new_V_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3,
                 V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

        # rospy.init_node("controller_outputs", anonymous=True)

        # self.pub_alpha = rospy.Publisher('/controller/alpha', Float64, queue_size=10)
        # self.pub_delta = rospy.Publisher('/controller/delta', Float64, queue_size=10)
        # self.pub_rho = rospy.Publisher('/controller/rho', Float64, queue_size=10)
        

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs:
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        rho = np.sqrt((x - self.x_g) ** 2 + (y - self.y_g) ** 2)
        alpha = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - th)
        delta = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - self.th_g)

        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * np.sinc(alpha / np.pi) * np.cos(alpha) * (alpha + self.k3 * delta)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om

class HeadingController:
    """
    pose stabilization controller
    """
    def __init__(self, kp, om_max=1):
        self.kp = kp
        self.om_max = om_max

    def load_goal(self, th_g):
        """
        loads in a new goal position
        """
        self.th_g = th_g

    def compute_control(self, x, y, th, t, prev_om=None):
        err = wrapToPi(self.th_g - th)
        om = self.kp*err

        if prev_om is not None:
            if np.abs(prev_om) <= 1e-3:
                om = np.clip(om, -0.2, 0.2)
            elif np.abs(prev_om) <= 0.5:
                om = np.clip(om, -1.5*np.abs(prev_om), 1.5*np.abs(prev_om))

        # apply control limits
        V = 0
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om