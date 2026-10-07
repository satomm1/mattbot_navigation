"""Where can an object be observed from, and where along a path should the robot stop to look.

Pure numpy (no ROS). Grids use the ROS OccupancyGrid layout: array[row, col] with
row = y index and col = x index counted from the map origin.

A viewshed is the set of map cells within [r_min, r_max] of an object that have an unobstructed
line of sight to it on the static map (and, optionally, past other known objects, which also
block the view). Line of sight is symmetric, so it is computed once per
object by casting rays outward from the object, not per robot cell. The viewshed holds every
viewpoint cell (on or off any path), so detour planning can search it later.
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


def compute_viewshed(
    blocking, obj_xy, origin, resolution, r_min, r_max, n_rays=None, ignore_near_m=0.15, blockers=()
):
    """Viewshed of an object at obj_xy (map frame, m) on the blocking mask.

    Rays are sampled every half cell and stop at the first blocking or off-map sample.
    Blocking cells within ignore_near_m of the object are ignored, since an object next to a
    wall (or partly drawn into the static map) would otherwise block every ray.
    blockers: other objects [(x, y, width), ...] whose square footprints also block rays.
    """
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


def select_stops(
    path_xy, targets, viewsheds, r_pref=2.0, skip_start_m=0.5, skip_goal_m=0.8, merge_m=0.5, max_turn=math.pi / 2
):
    """Choose where along path_xy (N, 2) to stop and observe each target.

    targets:   [(object_id, x, y), ...]
    viewsheds: {object_id: Viewshed}
    Each target gets at most one stop: the path point inside its viewshed whose distance to the
    object is closest to r_pref. Points from which the object is within max_turn (rad) of the
    path heading are preferred; points that need a larger turn (the object is behind, i.e. the
    stop overshoots it) are used only if there are no others. None = no preference. Points within skip_start_m (arc length) of the start or
    skip_goal_m of the goal are not used. Stops within merge_m of arc length are merged into
    the earlier one. Returns [Stop, ...] ordered along the path.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return []
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    to_goal = np.hypot(*(path_xy - path_xy[-1]).T)
    usable = (arc >= skip_start_m) & (to_goal >= skip_goal_m)

    headings = path_headings(path_xy)
    picks = []  # (path_index, object_id, x, y)
    for object_id, tx, ty in targets:
        vs = viewsheds.get(object_id)
        if vs is None:
            continue
        ok = usable & vs.contains_xy(path_xy)
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
