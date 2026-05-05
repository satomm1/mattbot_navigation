import numpy as np
import scipy.interpolate

from navigation_utils.trackers import TrajectoryTracker,  PoseController, HeadingController
from navigation_utils.grids import DetOccupancyGrid2D, StochOccupancyGrid2D
from navigation_utils.search import AStar
from navigation_utils.zone import load_zones_from_file, get_zone_for_point

def wrapToPi(a):
    if isinstance(a, list):
        return [(x+np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    assert(path and k > 2 and k < len(path))
    ########## Code starts here ##########
    t = np.zeros(len(path))
    for ii in range(len(path) - 1):
        dist = np.linalg.norm(np.array(path[ii + 1]) - np.array(path[ii]))  # distance between consecutive points
        t[ii + 1] = dist / V_des + t[ii]  # time at next point assuming constant velocity V_des

    tck_x = scipy.interpolate.splrep(t, np.array(path)[:, 0], k=k, s=alpha)
    tck_y = scipy.interpolate.splrep(t, np.array(path)[:, 1], k=k, s=alpha)

    t_smoothed = np.arange(0, t[-1], dt)
    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed


def compute_trajectory_from_timed_waypoints(path, t_waypoints, k, alpha, dt):
    """
    Like compute_smoothed_traj, but knot times in the time domain are given by t_waypoints
    (e.g. from a multi-agent timing MILP). path[i] is visited at time t_waypoints[i].

    path: sequence of (x, y), length N
    t_waypoints: length-N monotone non-decreasing times; typically t[0] == 0
    k, alpha, dt: same meaning as compute_smoothed_traj (splrep degree, smoothing, sample step)
    Returns:
        t_smoothed, traj_smoothed (N,7) as compute_smoothed_traj
    """
    assert path and k > 2 and k < len(path)
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be Nx2")
    t = np.asarray(t_waypoints, dtype=float).copy().reshape(-1)
    if t.shape[0] != path.shape[0]:
        raise ValueError("t_waypoints must have same length as path")
    t[0] = 0.0
    eps = 1e-4
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    t_end = float(t[-1])
    if t_end <= 0.0:
        raise ValueError("timed path must have positive duration (last t > 0)")

    tck_x = scipy.interpolate.splrep(t, path[:, 0], k=k, s=alpha)
    tck_y = scipy.interpolate.splrep(t, path[:, 1], k=k, s=alpha)

    t_smoothed = np.arange(0.0, t_end, float(dt))
    if t_smoothed.size == 0 or t_smoothed[-1] < t_end - 1e-9:
        t_smoothed = np.append(t_smoothed, t_end)

    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta_d = np.arctan2(yd_d, xd_d)
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()
    return t_smoothed, traj_smoothed

