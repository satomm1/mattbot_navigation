import numpy as np

V_EPS = 0.05  # minimum |V| for curvature (om) division at standstill

def wrapToPi(a):
    if isinstance(a, list):
        return [(x+np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """

    def __init__(self, kpx, kpy, kdx, kdy,
                 V_max=0.6, om_max=1, a_max=0.3, soft_start_sec=1.5,
                 hold_enter_speed=0.03, hold_exit_speed=0.08,
                 hold_pos_tol=0.08, hold_kp_pos=1.8, hold_kp_heading=2.5):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max
        self.a_max = a_max
        self.soft_start_sec = float(soft_start_sec)
        self.hold_enter_speed = float(hold_enter_speed)
        self.hold_exit_speed = float(hold_exit_speed)
        self.hold_pos_tol = float(hold_pos_tol)
        self.hold_kp_pos = float(hold_kp_pos)
        self.hold_kp_heading = float(hold_kp_heading)
        self._hold_active = False

        self.coeffs = np.zeros(8)  # Polynomial coefficients for x(t) and y(t) as
        # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        self._hold_active = False

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def _reference_time(self, t):
        """
        Map wall-clock track time to spline sample time.

        During soft start, reference time integrates the smoothstep speed scale so
        position/velocity stay consistent and the tracker does not fall behind then
        surge to catch up. After soft start, reference time trails wall clock by
        half of soft_start_sec (the lag accumulated during the ramp).
        """
        T = self.soft_start_sec
        t = max(0.0, float(t))
        if T <= 0.0:
            return t
        if t <= T:
            u = t / T
            # integral_0^u (3s^2 - 2s^3) ds = u^3 - u^4/2
            return T * (u * u * u - 0.5 * u * u * u * u)
        return t - 0.5 * T

    def get_desired_state(self, t):
        """
        Input:
            t: Current wall-clock time since TRACK began
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at reference time on the loaded trajectory
        """
        t_ref = self._reference_time(t)
        if len(self.traj_times) > 0:
            t_ref = min(t_ref, float(self.traj_times[-1]))

        x_d = np.interp(t_ref, self.traj_times, self.traj[:, 0])
        y_d = np.interp(t_ref, self.traj_times, self.traj[:, 1])
        xd_d = np.interp(t_ref, self.traj_times, self.traj[:, 3])
        yd_d = np.interp(t_ref, self.traj_times, self.traj[:, 4])
        xdd_d = np.interp(t_ref, self.traj_times, self.traj[:, 5])
        ydd_d = np.interp(t_ref, self.traj_times, self.traj[:, 6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def _V_for_om(self, V, v_ref=0.0):
        """Denominator for curvature; use ref speed so om does not spike at V≈0."""
        return max(abs(V), abs(v_ref), V_EPS)

    def _apply_accel_slew(self, V_cmd, V_prev, dt):
        """Limit rate of increase of |V|; deceleration is not slew-limited."""
        if dt <= 0.0:
            return V_prev

        if abs(V_cmd) <= abs(V_prev) + 1e-9:
            return V_cmd

        if np.sign(V_cmd) == np.sign(V_prev) or abs(V_prev) < 1e-3:
            V_mag = min(abs(V_cmd), abs(V_prev) + self.a_max * dt)
            return np.sign(V_cmd) * V_mag

        return V_cmd

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = max(t - self.t_prev, 0.0)
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)
        v_ref = np.hypot(xd_d, yd_d)

        # Hold-aware mode: for scheduled waits (near-zero reference speed), stop at the
        # hold point instead of coasting through and later correcting with reverse motion.
        if self._hold_active:
            self._hold_active = v_ref < self.hold_exit_speed
        else:
            self._hold_active = v_ref < self.hold_enter_speed

        if self._hold_active:
            err_x = x_d - x
            err_y = y_d - y
            rho = np.hypot(err_x, err_y)
            if rho <= self.hold_pos_tol:
                V = 0.0
                om = 0.0
            else:
                alpha = wrapToPi(np.arctan2(err_y, err_x) - th)
                # Forward-only approach while holding; rotate in place if target is behind.
                V_target = self.hold_kp_pos * rho * max(0.0, np.cos(alpha))
                V_target = min(V_target, 0.35 * self.V_max)
                om = self.hold_kp_heading * alpha
                V = self._apply_accel_slew(V_target, self.V_prev, dt)
        else:
            ########## Code starts here ##########
            V_div = self._V_for_om(self.V_prev, v_ref)

            x_dot = self.V_prev * np.cos(th)
            y_dot = self.V_prev * np.sin(th)

            u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
            u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

            a = u1 * np.cos(th) + u2 * np.sin(th)
            om = (-u1 * np.sin(th) + u2 * np.cos(th)) / V_div

            V = self.V_prev + a * dt if dt > 0.0 else self.V_prev
            V = self._apply_accel_slew(V, self.V_prev, dt)
            ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # If near the end of the trajectory, slow down so we don't stop abruptly
        t_end = float(self.traj_times[-1])
        x_goal = np.interp(t_end, self.traj_times, self.traj[:, 0])
        y_goal = np.interp(t_end, self.traj_times, self.traj[:, 1])
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
    Spin-in-place heading controller using trapezoidal deceleration.

    - Far from goal: |omega| = om_max (cruise).
    - Near goal: |omega| = min(om_max, sqrt(2 * alpha_max * |err|)) so the robot can stop in remaining angle.
    - Outside the stop zone (|err| > err_stop): enforce om_min so commands avoid stiction twitch.
    - Slew limit (alpha_max * dt) applies only when increasing |omega| (acceleration).
    """

    def __init__(self, om_max=1, om_min=1.0, alpha_max=2.0, err_deadband=0.02):
        self.om_max = om_max
        self.om_min = om_min
        self.alpha_max = alpha_max
        self.err_deadband = err_deadband
        # Below this |err|, om_min is not applied (trapezoid may command slower approach).
        self.err_stop = (om_min ** 2) / (2.0 * alpha_max)
        self.t_prev = None

    def load_goal(self, th_g):
        """Load a new goal heading and reset timing state."""
        self.th_g = th_g
        self.t_prev = None

    def _trapezoid_omega(self, err):
        """Trapezoidal spin profile with optional minimum |omega| when moving."""
        if abs(err) < self.err_deadband:
            return 0.0

        om_mag = min(self.om_max, np.sqrt(2.0 * self.alpha_max * abs(err)))
        if abs(err) > self.err_stop:
            om_mag = max(self.om_min, om_mag)

        return np.sign(err) * om_mag

    def _apply_accel_slew(self, om_cmd, prev_om, dt):
        """Limit rate of increase of |omega|; deceleration is not slew-limited."""
        if prev_om is None or dt <= 0.0:
            return om_cmd

        if abs(om_cmd) <= abs(prev_om) + 1e-9:
            return om_cmd

        # Only limit when magnitude increases with the same sign (or from rest).
        if np.sign(om_cmd) == np.sign(prev_om) or abs(prev_om) < 1e-3:
            om_mag = min(abs(om_cmd), abs(prev_om) + self.alpha_max * dt)
            return np.sign(om_cmd) * om_mag

        return om_cmd

    def compute_control(self, th, t, prev_om=None):
        err = wrapToPi(self.th_g - th)

        if t is not None and self.t_prev is not None:
            dt = max(float(t - self.t_prev), 1e-3)
        else:
            dt = 0.1
        if t is not None:
            self.t_prev = t

        om = self._trapezoid_omega(err)
        om = self._apply_accel_slew(om, prev_om, dt)

        V = 0
        om = np.clip(om, -self.om_max, self.om_max)
        return V, om


class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1.5):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max       

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

        """
        rho = np.sqrt((x - self.x_g) ** 2 + (y - self.y_g) ** 2)
        alpha = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - th)
        delta = wrapToPi(np.arctan2(self.y_g - y, self.x_g - x) - self.th_g)

        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * np.sinc(alpha / np.pi) * np.cos(alpha) * (alpha + self.k3 * delta)

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
