"""Where can an object be observed from, and where along a path should the robot stop to look.

Pure numpy (no ROS). Grids use the ROS OccupancyGrid layout: array[row, col] with
row = y index and col = x index counted from the map origin.

A viewshed is the set of map cells within [r_min, r_max] of an object that have an unobstructed
line of sight to it on the static map (and, optionally, past other known objects, which also
block the view). Line of sight is symmetric, so it is computed once per
object by casting rays outward from the object, not per robot cell. The viewshed holds every
viewpoint cell (on or off any path), so detour planning can search it later.

With the object's width, a cell is a viewpoint only if most of the object is in view from it
(LOS_OFFSETS / LOS_MIN_CLEAR): a cell where a wall corner hides most of the object is not one, so
no stop, detour or scout looks from there (the evaluator would read the hidden object as gone).
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def blocking_mask(occupancy_data, width, height, occupied_thresh=50):
    """(height, width) bool: cells that block line of sight (occupied or unknown)."""
    grid = np.asarray(occupancy_data, dtype=np.int16).reshape(height, width)
    return (grid >= occupied_thresh) | (grid < 0)


@dataclass
class Viewshed:
    obj_xy: Tuple[float, float]
    i0: int  # map column of mask[:, 0]
    j0: int  # map row of mask[0, :]
    mask: np.ndarray  # bool (rows, cols)
    origin: Tuple[float, float]
    resolution: float

    def contains_xy(self, xy):
        """Bool per point of xy (N, 2): is the point's cell a viewpoint of the object."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        i = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int) - self.i0
        j = np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(int) - self.j0
        rows, cols = self.mask.shape
        inside = (i >= 0) & (i < cols) & (j >= 0) & (j < rows)
        out = np.zeros(len(xy), dtype=bool)
        out[inside] = self.mask[j[inside], i[inside]]
        return out

    def num_cells(self):
        return int(self.mask.sum())


MIN_BLOCKER_M = 0.2  # smallest footprint side for a known object blocking line of sight

# Line of sight to an object of some width: rays from the viewpoint to points across the object (fractions of its
# radius, perpendicular to the view), each ending at the object's near side. The object is in view if at least
# LOS_MIN_CLEAR of them are unblocked: a detector still finds a mostly visible object, so a thin occluder (a chair
# leg, one stray map cell) does not hide it, while a wall corner hiding most of it does. The sim's camera
# (mattbot_sim World.line_of_sight) uses the same rule.
LOS_OFFSETS = (-0.8, -0.4, 0.0, 0.4, 0.8)
LOS_MIN_CLEAR = 3
LOS_SURFACE_MARGIN_M = 0.05  # rays stop this short of the object's surface
# A viewpoint must pass the rule from every cell within this distance: the robot never stops exactly on the planned
# cell (tracking and localization error), and near a wall corner visibility changes within a few centimetres.
LOS_POSE_TOLERANCE_M = 0.1


def compute_viewshed(
    blocking, obj_xy, origin, resolution, r_min, r_max, n_rays=None, ignore_near_m=0.15, blockers=(), obj_width=None
):
    """Viewshed of an object at obj_xy (map frame, m) on the blocking mask.

    Rays are sampled every half cell and stop at the first blocking or off-map sample.
    Blocking cells within ignore_near_m of the object are ignored, since an object next to a
    wall (or partly drawn into the static map) would otherwise block every ray.
    blockers: other objects [(x, y, width), ...] whose square footprints also block rays.
    obj_width: the object's width (m). Given, a cell is kept only if at least LOS_MIN_CLEAR of the
    LOS_OFFSETS rays across the object are clear from the centre of every free cell within
    LOS_POSE_TOLERANCE_M of it (see LOS_OFFSETS); without it, one ray to the object's centre decides.
    """
    if obj_width is not None and obj_width > 0.0:
        return _width_viewshed(blocking, obj_xy, origin, resolution, r_min, r_max, ignore_near_m, blockers,
                               0.5 * float(obj_width))
    height, width = blocking.shape
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    step = resolution / 2.0
    if n_rays is None:
        # Keep the gap between neighbouring rays at r_max below half a cell
        n_rays = max(360, int(math.ceil(2.0 * math.pi * r_max / step)))

    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    dists = np.arange(step, r_max + step, step)
    xs = ox + np.cos(angles)[:, None] * dists[None, :]
    ys = oy + np.sin(angles)[:, None] * dists[None, :]
    ci = np.floor((xs - origin[0]) / resolution).astype(int)
    cj = np.floor((ys - origin[1]) / resolution).astype(int)
    on_map = (ci >= 0) & (ci < width) & (cj >= 0) & (cj < height)

    hit = ~on_map
    hit[on_map] = blocking[cj[on_map], ci[on_map]]
    for bx, by, bw in blockers:
        half = max(bw, MIN_BLOCKER_M) / 2.0
        if math.hypot(bx - ox, by - oy) > r_max + half:
            continue  # cannot reach any ray
        hit |= (np.abs(xs - bx) <= half) & (np.abs(ys - by) <= half)
    hit[:, dists < ignore_near_m] = False
    blocked = np.cumsum(hit, axis=1) > 0  # everything from the first hit outward
    visible = ~blocked & on_map & (dists >= r_min)[None, :]

    # Bounding box of the r_max disc, clipped to the map
    i0 = max(int(math.floor((ox - r_max - origin[0]) / resolution)), 0)
    j0 = max(int(math.floor((oy - r_max - origin[1]) / resolution)), 0)
    i1 = min(int(math.floor((ox + r_max - origin[0]) / resolution)), width - 1)
    j1 = min(int(math.floor((oy + r_max - origin[1]) / resolution)), height - 1)
    # Samples at exactly r_max can round into the cell just outside the box
    visible &= (ci >= i0) & (ci <= i1) & (cj >= j0) & (cj <= j1)
    mask = np.zeros((max(j1 - j0 + 1, 0), max(i1 - i0 + 1, 0)), dtype=bool)
    if mask.size:
        mask[cj[visible] - j0, ci[visible] - i0] = True
    return Viewshed((ox, oy), i0, j0, mask, (float(origin[0]), float(origin[1])), float(resolution))


def _width_viewshed(blocking, obj_xy, origin, resolution, r_min, r_max, ignore_near_m, blockers, radius):
    """compute_viewshed with obj_width: the width rule from every cell centre, then the pose tolerance."""
    height, width = blocking.shape
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    i0 = max(int(math.floor((ox - r_max - origin[0]) / resolution)), 0)
    j0 = max(int(math.floor((oy - r_max - origin[1]) / resolution)), 0)
    i1 = min(int(math.floor((ox + r_max - origin[0]) / resolution)), width - 1)
    j1 = min(int(math.floor((oy + r_max - origin[1]) / resolution)), height - 1)
    shape = (max(j1 - j0 + 1, 0), max(i1 - i0 + 1, 0))
    if not (shape[0] and shape[1]):
        return Viewshed((ox, oy), i0, j0, np.zeros(shape, dtype=bool), (float(origin[0]), float(origin[1])),
                        float(resolution))
    # Points within the tolerance lie in cells whose centres are within it plus half a cell diagonal
    reach = LOS_POSE_TOLERANCE_M + resolution * math.sqrt(0.5)
    tol = int(math.floor(reach / resolution + 1e-9))
    # Test a margin of tol cells round the box too, so the tolerance sees the cells just outside it
    ja, jb = max(j0 - tol, 0), min(j1 + tol, height - 1)
    ia, ib = max(i0 - tol, 0), min(i1 + tol, width - 1)
    jj, ii = np.mgrid[ja:jb + 1, ia:ib + 1]
    cx = origin[0] + (ii + 0.5) * resolution  # cell centres
    cy = origin[1] + (jj + 0.5) * resolution
    dist = np.hypot(ox - cx, oy - cy)
    walls = blocking[jj, ii]
    test = (dist >= r_min - reach - resolution) & (dist <= r_max + reach + resolution) & ~walls
    seen = np.zeros(jj.shape, dtype=bool)
    seen[test] = width_line_of_sight_counts(blocking, origin, resolution, cx[test], cy[test], ox, oy, radius,
                                            ignore_near_m, blockers) >= LOS_MIN_CLEAR
    # Every free cell within tol cells must see it (walls: the robot is never there); off-map counts as a wall
    ok = seen | walls
    padded = np.pad(ok, tol, constant_values=True)
    robust = seen.copy()
    for dj in range(-tol, tol + 1):
        for di in range(-tol, tol + 1):
            if (dj or di) and (dj * dj + di * di) * resolution * resolution <= reach ** 2 + 1e-12:
                robust &= padded[tol + dj:tol + dj + ok.shape[0], tol + di:tol + di + ok.shape[1]]
    robust &= (dist >= r_min) & (dist <= r_max)
    mask = robust[j0 - ja:j0 - ja + shape[0], i0 - ia:i0 - ia + shape[1]]
    return Viewshed((ox, oy), i0, j0, np.ascontiguousarray(mask), (float(origin[0]), float(origin[1])),
                    float(resolution))


def width_line_of_sight_counts(blocking, origin, resolution, cx, cy, ox, oy, radius, ignore_near_m=0.15, blockers=()):
    """Per viewpoint (cx, cy arrays): how many LOS_OFFSETS rays reach the object (centre ox, oy, radius) unblocked.

    Off-map samples block. Blocking cells within ignore_near_m of the object centre are ignored (see
    compute_viewshed); blockers [(x, y, width)] block with their square footprints."""
    height, width = blocking.shape
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    vx, vy = ox - cx, oy - cy
    dist = np.maximum(np.hypot(vx, vy), 1e-9)
    px, py = -vy / dist, vx / dist  # perpendicular to the view
    step = resolution / 2.0
    n_steps = int(math.ceil((float(dist.max()) if dist.size else 0.0) / step)) + 1
    s = np.arange(1, n_steps + 1) * step
    near_blockers = [(bx, by, max(bw, MIN_BLOCKER_M) / 2.0) for bx, by, bw in blockers]
    clear = np.zeros(cx.shape, dtype=np.int64)
    for f in LOS_OFFSETS:
        tx, ty = ox + f * radius * px - cx, oy + f * radius * py - cy  # to the point across the object
        tlen = np.maximum(np.hypot(tx, ty), 1e-9)
        ux, uy = tx / tlen, ty / tlen
        # where the ray enters the object's disc (as world.ray_circle), capped at the point itself
        t = vx * ux + vy * uy
        disc = radius * radius - (vx * vx + vy * vy - t * t)
        entry = np.where(disc >= 0.0, t - np.sqrt(np.maximum(disc, 0.0)), tlen)
        reach = np.minimum(entry, tlen) - LOS_SURFACE_MARGIN_M
        xs = cx[:, None] + ux[:, None] * s[None, :]
        ys = cy[:, None] + uy[:, None] * s[None, :]
        live = s[None, :] < reach[:, None] + step  # samples as mattbot_sim World.raycast takes them (up to reach + step)
        ci = np.floor((xs - origin[0]) / resolution).astype(np.int64)
        cj = np.floor((ys - origin[1]) / resolution).astype(np.int64)
        on_map = (ci >= 0) & (ci < width) & (cj >= 0) & (cj < height)
        hit = ~on_map
        hit[on_map] = blocking[cj[on_map], ci[on_map]]
        hit &= np.hypot(xs - ox, ys - oy) >= ignore_near_m
        for bx, by, half in near_blockers:
            hit |= (np.abs(xs - bx) <= half) & (np.abs(ys - by) <= half)
        clear += ~np.any(hit & live, axis=1) | (reach <= 0.0)
    return clear


def nearby_blockers(target_id, target_xy, objects, radius):
    """Other objects within radius of the target that may block its line of sight.

    objects: [(object_id, x, y, width), ...]. Returns (blockers, signature): blockers as
    [(x, y, width)], and a hashable signature (rounded to 0.1 m) for cache invalidation.
    """
    near = sorted(
        (oid, x, y, w)
        for oid, x, y, w in objects
        if oid != target_id and math.hypot(x - target_xy[0], y - target_xy[1]) <= radius
    )
    blockers = [(x, y, w) for _oid, x, y, w in near]
    signature = tuple((oid, round(x, 1), round(y, 1), round(w, 1)) for oid, x, y, w in near)
    return blockers, signature


@dataclass
class Stop:
    """Place on the path to stop and look at one or more objects."""

    path_index: int
    x: float
    y: float
    heading: float  # path heading at the stop (to turn back to)
    targets: List[Tuple[str, float, float]] = field(default_factory=list)  # (object_id, x, y), turn order


def path_headings(path_xy):
    """Heading of the path at each point (central differences)."""
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return np.zeros(len(path_xy))
    d = np.gradient(path_xy, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def usable_points(path_xy, skip_start_m=0.5, skip_goal_m=0.8):
    """Bool per point of path_xy (N, 2): may an on-path stop be placed there. Points within
    skip_start_m (arc length) of the start or skip_goal_m (straight line) of the goal are not usable."""
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return np.zeros(len(path_xy), dtype=bool)
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    to_goal = np.hypot(*(path_xy - path_xy[-1]).T)
    return (arc >= skip_start_m) & (to_goal >= skip_goal_m)


def select_stops(
    path_xy, targets, viewsheds, r_pref=2.0, skip_start_m=0.5, skip_goal_m=0.8, merge_m=0.5, max_turn=math.pi / 2,
    due_masks=None,
):
    """Choose where along path_xy (N, 2) to stop and observe each target.

    targets:   [(object_id, x, y), ...]
    viewsheds: {object_id: Viewshed}
    Each target gets at most one stop: the path point inside its viewshed whose distance to the
    object is closest to r_pref. Points from which the object is within max_turn (rad) of the
    path heading are preferred; points that need a larger turn (the object is behind, i.e. the
    stop overshoots it) are used only if there are no others. None = no preference. Points within skip_start_m (arc length) of the start or
    skip_goal_m of the goal are not used. Stops within merge_m of arc length are merged into
    the earlier one. due_masks {object_id: bool per path point}: only points where the object is due
    (e.g. its belief will have decayed by the time the robot gets there) may be used.
    Returns [Stop, ...] ordered along the path.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return []
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    usable = usable_points(path_xy, skip_start_m, skip_goal_m)

    headings = path_headings(path_xy)
    picks = []  # (path_index, object_id, x, y)
    for object_id, tx, ty in targets:
        vs = viewsheds.get(object_id)
        if vs is None:
            continue
        ok = usable & vs.contains_xy(path_xy)
        if due_masks is not None and object_id in due_masks:
            ok &= due_masks[object_id]
        if not ok.any():
            continue
        if max_turn is not None:
            bearing = np.arctan2(ty - path_xy[:, 1], tx - path_xy[:, 0])
            turn = np.abs((bearing - headings + np.pi) % (2.0 * np.pi) - np.pi)
            small_turn = ok & (turn <= max_turn)
            if small_turn.any():
                ok = small_turn
        dist = np.hypot(path_xy[:, 0] - tx, path_xy[:, 1] - ty)
        score = np.where(ok, np.abs(dist - r_pref), np.inf)
        picks.append((int(np.argmin(score)), object_id, float(tx), float(ty)))
    picks.sort(key=lambda p: (p[0], p[1]))

    stops = []
    for idx, object_id, tx, ty in picks:
        if stops and arc[idx] - arc[stops[-1].path_index] <= merge_m:
            stops[-1].targets.append((object_id, tx, ty))
            continue
        stops.append(Stop(idx, float(path_xy[idx, 0]), float(path_xy[idx, 1]), float(headings[idx]),
                          [(object_id, tx, ty)]))

    # Sweep targets in one direction: order by signed angle relative to the path heading
    for s in stops:
        s.targets.sort(key=lambda t: wrap_angle(math.atan2(t[2] - s.y, t[1] - s.x) - s.heading))
    return stops
