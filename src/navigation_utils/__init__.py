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


def plan_start_heading(unsmoothed_plan, traj_new, v_min=0.05):
    """
    Heading target for ALIGN before TRACK.

    Prefer atan2(yd, xd) from the smoothed trajectory (matches TRACK departure).
    Fall back to the first grid segment bearing, then traj_new[0, 2].
    """
    if traj_new is not None and len(traj_new) > 0:
        traj_new = np.asarray(traj_new)
        if traj_new.ndim == 2 and traj_new.shape[1] >= 5:
            for i in range(traj_new.shape[0]):
                xd = float(traj_new[i, 3])
                yd = float(traj_new[i, 4])
                if xd * xd + yd * yd >= v_min * v_min:
                    return float(np.arctan2(yd, xd))
        if traj_new.ndim == 2 and traj_new.shape[1] >= 3:
            return float(traj_new[0, 2])

    plan = unsmoothed_plan
    if plan is not None and len(plan) >= 2:
        dx = float(plan[1][0]) - float(plan[0][0])
        dy = float(plan[1][1]) - float(plan[0][1])
        if dx * dx + dy * dy >= 1e-8:
            return float(np.arctan2(dy, dx))

    return 0.0


def _turn_angle_rad(path, i):
    """Interior turn angle (rad) at vertex path[i]; 0 = straight, pi = U-turn."""
    v_in = path[i] - path[i - 1]
    v_out = path[i + 1] - path[i]
    n_in = np.linalg.norm(v_in)
    n_out = np.linalg.norm(v_out)
    if n_in < 1e-9 or n_out < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
    return float(np.arccos(cos))


def find_sharp_corner_indices(path, min_turn_deg=45.0):
    """Indices of interior vertices where the polyline turns at least min_turn_deg."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3:
        return []
    min_angle = np.deg2rad(float(min_turn_deg))
    corners = []
    for i in range(1, path.shape[0] - 1):
        if _turn_angle_rad(path, i) >= min_angle:
            corners.append(i)
    return corners


def _leg_inset_points(p_prev, p_corner, p_next, inset_m, points_per_leg):
    """Polyline points inset along incoming/outgoing legs toward a sharp corner."""
    p_prev = np.asarray(p_prev, dtype=float)
    p_corner = np.asarray(p_corner, dtype=float)
    p_next = np.asarray(p_next, dtype=float)
    incoming = []
    outgoing = []

    v_in = p_corner - p_prev
    len_in = float(np.linalg.norm(v_in))
    if len_in > 1e-9 and points_per_leg > 0:
        dir_in = v_in / len_in
        max_in = min(float(inset_m), len_in * 0.45)
        for k in range(points_per_leg, 0, -1):
            incoming.append(p_corner - (k / points_per_leg) * max_in * dir_in)

    v_out = p_next - p_corner
    len_out = float(np.linalg.norm(v_out))
    if len_out > 1e-9 and points_per_leg > 0:
        dir_out = v_out / len_out
        max_out = min(float(inset_m), len_out * 0.45)
        for k in range(1, points_per_leg + 1):
            outgoing.append(p_corner + (k / points_per_leg) * max_out * dir_out)

    return incoming, outgoing


def _time_at_point_on_segment(p0, p1, t0, t1, p):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p = np.asarray(p, dtype=float)
    seg = p1 - p0
    len2 = float(seg @ seg)
    if len2 < 1e-18:
        return float(t0)
    frac = float(np.clip((p - p0) @ seg / len2, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0))


def _dedupe_consecutive_points(path, times=None, min_dist=1e-6):
    path = np.asarray(path, dtype=float)
    if path.size == 0:
        return path, times
    keep = [0]
    for i in range(1, path.shape[0]):
        if np.linalg.norm(path[i] - path[keep[-1]]) >= min_dist:
            keep.append(i)
    path_out = path[keep]
    if times is None:
        return path_out, None
    times = np.asarray(times, dtype=float).reshape(-1)
    return path_out, times[keep]


def insert_corner_interpolation_points(
    path,
    min_turn_deg=45.0,
    inset_m=0.10,
    points_per_leg=3,
    t_waypoints=None,
):
    """
    Densify a grid path at sharp corners so splines stay near the polyline.

    At each vertex whose turn angle is at least min_turn_deg, inserts points_per_leg
    waypoints along the incoming and outgoing legs (within inset_m of the corner).
    When t_waypoints is provided, times are interpolated linearly along each leg.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be Nx2")
    if path.shape[0] < 3:
        out = path
        if t_waypoints is not None:
            return out, np.asarray(t_waypoints, dtype=float).reshape(-1)
        return out

    corners = set(find_sharp_corner_indices(path, min_turn_deg))
    if not corners:
        if t_waypoints is not None:
            return path, np.asarray(t_waypoints, dtype=float).reshape(-1)
        return path

    timed = t_waypoints is not None
    if timed:
        t_waypoints = np.asarray(t_waypoints, dtype=float).reshape(-1)
        if t_waypoints.shape[0] != path.shape[0]:
            raise ValueError("t_waypoints must have same length as path")

    out_pts = []
    out_times = [] if timed else None
    for i in range(path.shape[0]):
        if i in corners:
            incoming, outgoing = _leg_inset_points(
                path[i - 1], path[i], path[i + 1], inset_m, points_per_leg
            )
            for pt in incoming:
                out_pts.append(pt)
                if timed:
                    out_times.append(_time_at_point_on_segment(path[i - 1], path[i], t_waypoints[i - 1], t_waypoints[i], pt))
        out_pts.append(path[i])
        if timed:
            out_times.append(t_waypoints[i])
        if i in corners:
            for pt in outgoing:
                out_pts.append(pt)
                if timed:
                    out_times.append(_time_at_point_on_segment(path[i], path[i + 1], t_waypoints[i], t_waypoints[i + 1], pt))
    out = np.asarray(out_pts, dtype=float)
    if timed:
        out, out_t = _dedupe_consecutive_points(out, np.asarray(out_times, dtype=float))
        return out, out_t
    return _dedupe_consecutive_points(out, None)[0]


def compute_smoothed_traj(path, V_des, k, alpha, dt,
                          corner_aware=True, corner_min_turn_deg=45.0,
                          corner_inset_m=0.10, corner_points_per_leg=3):
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
    path = np.asarray(path, dtype=float)
    if corner_aware and path.shape[0] >= 3:
        path = insert_corner_interpolation_points(
            path,
            min_turn_deg=corner_min_turn_deg,
            inset_m=corner_inset_m,
            points_per_leg=corner_points_per_leg,
        )
    assert(path is not None and len(path) > 0 and k > 2 and k < len(path))
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


def _cumulative_arc_length_xy(xy):
    """Cumulative Euclidean arc length along rows of an Nx2 polyline (length N)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be Nx2")
    if xy.shape[0] < 1:
        return np.zeros(0, dtype=float)
    if xy.shape[0] == 1:
        return np.zeros(1, dtype=float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _enforce_monotone_waypoint_times(t_waypoints, eps=1e-4):
    """Force t[0]==0 and strictly non-decreasing knot times (for MILP schedules)."""
    t = np.asarray(t_waypoints, dtype=float).copy().reshape(-1)
    if t.size == 0:
        return t
    t[0] = 0.0
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def _prepare_timed_path_and_schedule(path, t_waypoints, corner_aware, corner_min_turn_deg,
                                     corner_inset_m, corner_points_per_leg):
    """
    Corner-densify path (same as compute_smoothed_traj) and build arc-length + time schedules.

    Returns (path, t_wp, s_wp) with len(t_wp)==len(s_wp)==len(path).
    """
    path = np.asarray(path, dtype=float)
    t_wp = np.asarray(t_waypoints, dtype=float).reshape(-1)
    if corner_aware and path.shape[0] >= 3:
        path, t_wp = insert_corner_interpolation_points(
            path,
            min_turn_deg=corner_min_turn_deg,
            inset_m=corner_inset_m,
            points_per_leg=corner_points_per_leg,
            t_waypoints=t_wp,
        )
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be Nx2")
    if t_wp.shape[0] != path.shape[0]:
        raise ValueError("t_waypoints must have same length as path")
    t_wp = _enforce_monotone_waypoint_times(t_wp)
    s_wp = _cumulative_arc_length_xy(path)
    if float(s_wp[-1]) <= 0.0:
        raise ValueError("timed path must have positive arc length")
    if float(t_wp[-1]) <= 0.0:
        raise ValueError("timed path must have positive duration (last t > 0)")
    return path, t_wp, s_wp


def compute_trajectory_from_timed_waypoints(path, t_waypoints, k, alpha, dt,
                                            corner_aware=True, corner_min_turn_deg=45.0,
                                            corner_inset_m=0.10, corner_points_per_leg=3):
    """
    Build a wall-clock trajectory from a grid path and per-waypoint arrival times (e.g. MILP).

    Geometry and timing are decoupled:
      - Spatial path x(s), y(s) is a cubic spline in **arc length** s (same corner densification
        and splrep settings as compute_smoothed_traj).
      - Schedule s(t) is piecewise-linear between (t_waypoints[i], s_i). Long MILP waits become
        slow forward motion or hold along the same curve, not spatial loops.

    Single-robot navigation should keep using compute_smoothed_traj (distance-based knot times).

    path: sequence of (x, y), length N
    t_waypoints: length-N monotone non-decreasing times; typically t[0] == 0
    k, alpha, dt: same meaning as compute_smoothed_traj
    Returns:
        t_smoothed, traj_smoothed (N,7) compatible with TrajectoryTracker.load_traj
    """
    assert path and k > 2 and k < len(path)
    path, t_wp, s_wp = _prepare_timed_path_and_schedule(
        path,
        t_waypoints,
        corner_aware,
        corner_min_turn_deg,
        corner_inset_m,
        corner_points_per_leg,
    )
    t_end = float(t_wp[-1])

    # Geometry: splrep in arc length (not wall-clock MILP times).
    tck_x = scipy.interpolate.splrep(s_wp, path[:, 0], k=k, s=alpha)
    tck_y = scipy.interpolate.splrep(s_wp, path[:, 1], k=k, s=alpha)

    t_smoothed = np.arange(0.0, t_end, float(dt))
    if t_smoothed.size == 0 or t_smoothed[-1] < t_end - 1e-9:
        t_smoothed = np.append(t_smoothed, t_end)

    # Schedule s(t) from MILP waypoint times; hold/slow segments when delta t >> delta s.
    s_ref = np.interp(t_smoothed, t_wp, s_wp)
    ds_dt = np.gradient(s_ref, t_smoothed)
    d2s_dt2 = np.gradient(ds_dt, t_smoothed)

    dx_ds = scipy.interpolate.splev(s_ref, tck_x, der=1)
    dy_ds = scipy.interpolate.splev(s_ref, tck_y, der=1)
    d2x_ds2 = scipy.interpolate.splev(s_ref, tck_x, der=2)
    d2y_ds2 = scipy.interpolate.splev(s_ref, tck_y, der=2)

    x_d = scipy.interpolate.splev(s_ref, tck_x, der=0)
    y_d = scipy.interpolate.splev(s_ref, tck_y, der=0)
    xd_d = dx_ds * ds_dt
    yd_d = dy_ds * ds_dt
    xdd_d = d2x_ds2 * (ds_dt ** 2) + dx_ds * d2s_dt2
    ydd_d = d2y_ds2 * (ds_dt ** 2) + dy_ds * d2s_dt2
    theta_d = np.arctan2(yd_d, xd_d)
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()
    return t_smoothed, traj_smoothed
