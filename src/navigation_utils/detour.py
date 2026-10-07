"""Extra driving distance to observe an object that cannot be seen from the planned path.

Pure Python (numpy, scipy; no ROS). Works on the roadmap's coarse C-space grid (roadmap.cfree,
0.2 m): cells the robot centre can occupy, 8-connected.

Detour definition (leave anywhere, replan to the goal): the robot follows its path P (length L)
to some point P_i, drives to a viewpoint v of the object, then replans from v to the goal g:

    detour(v) = min_i (arc_i + d(P_i, v)) + d(v, g) - L
              = F(v) + G(v) - L

F is one Dijkstra from a virtual source joined to every path cell P_i with weight arc_i (the
distance driven along the path before leaving it), G one Dijkstra from the goal. Both are
bounded by L + max_detour_m. An object's detour is the minimum over its viewpoints (its viewshed
from viewpoints.compute_viewshed, mapped to C-space cells), so one path query costs two grid
Dijkstras for all objects together. L is taken as F(g) so that an on-path viewpoint costs 0.

Far objects are skipped before any search with a lower bound from straight-line distances:
    detour >= min_i (arc_i + |P_i - o|) + |o - g| - L - 2 r_max
(every viewpoint lies within r_max of the object o).
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class DetourParams:
    max_detour_m: float = 15.0  # objects whose cheapest detour is longer are ignored
    r_max: float = 3.5  # m; viewshed radius (for the far-object lower bound)
    snap_radius_m: float = 0.5  # path points off C-space snap to a free cell this close
    on_path_m: float = 0.3  # detours shorter than this count as "visible from the path"


@dataclass
class DetourResult:
    object_id: str
    detour_m: float  # extra distance vs. the planned path
    viewpoint_xy: Tuple[float, float]  # where to look from (coarse cell centre)
    leave_index: int  # index into the request path where the robot leaves it
    leave_xy: Tuple[float, float]
    to_view_m: float  # distance from the leave point to the viewpoint
    arrival_heading: float  # heading when reaching the viewpoint (rad)
    route_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # leave point -> viewpoint

    @property
    def on_path(self):
        return self.detour_m <= 0.0

    def to_dict(self):
        return {
            "detour_m": self.detour_m,
            "viewpoint": list(self.viewpoint_xy),
            "leave_index": self.leave_index,
            "leave": list(self.leave_xy),
            "to_view_m": self.to_view_m,
        }


class DetourPlanner:
    """Grid graph over a roadmap's C-space; built once per roadmap."""

    def __init__(self, roadmap):
        self.roadmap = roadmap
        free = roadmap.cfree
        h, w = free.shape
        self.cells = np.argwhere(free)  # node -> (row, col)
        self.n = len(self.cells)
        self.node_of = -np.ones(free.shape, dtype=np.int64)
        self.node_of[free] = np.arange(self.n)
        self.xy = np.stack(
            [roadmap.origin[0] + (self.cells[:, 1] + 0.5) * roadmap.res,
             roadmap.origin[1] + (self.cells[:, 0] + 0.5) * roadmap.res], axis=1)
        rows, cols, wts = [], [], []
        for dr, dc in _STEPS:
            r2, c2 = self.cells[:, 0] + dr, self.cells[:, 1] + dc
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            ok[ok] = free[r2[ok], c2[ok]]
            rows.append(self.node_of[self.cells[ok, 0], self.cells[ok, 1]])
            cols.append(self.node_of[r2[ok], c2[ok]])
            wts.append(np.full(int(ok.sum()), math.hypot(dr, dc) * roadmap.res))
        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._wts = np.concatenate(wts)
        self.graph = csr_matrix((self._wts, (self._rows, self._cols)), shape=(self.n, self.n))
        self._tree = cKDTree(self.xy) if self.n else None
        self.num_searches = 0  # path queries that ran the Dijkstra searches (diagnostics / tests)

    # ---------- Viewpoints ----------

    def viewpoint_nodes(self, viewshed):
        """C-space nodes whose cell contains at least one viewshed cell (viewpoints.Viewshed)."""
        jj, ii = np.nonzero(viewshed.mask)
        if len(jj) == 0:
            return np.zeros(0, dtype=np.int64)
        x = viewshed.origin[0] + (viewshed.i0 + ii + 0.5) * viewshed.resolution
        y = viewshed.origin[1] + (viewshed.j0 + jj + 0.5) * viewshed.resolution
        rm = self.roadmap
        r = np.floor((y - rm.origin[1]) / rm.res).astype(np.int64)
        c = np.floor((x - rm.origin[0]) / rm.res).astype(np.int64)
        h, w = rm.shape
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        nodes = self.node_of[r[ok], c[ok]]
        return np.unique(nodes[nodes >= 0])

    # ---------- Path ----------

    def _path_nodes(self, path_xy, snap_radius_m):
        """(node per path point or -1, metric arc length per path point)."""
        path_xy = np.asarray(path_xy, dtype=float)
        seg = np.hypot(*np.diff(path_xy, axis=0).T) if len(path_xy) > 1 else np.zeros(0)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        rm = self.roadmap
        r = np.floor((path_xy[:, 1] - rm.origin[1]) / rm.res).astype(np.int64)
        c = np.floor((path_xy[:, 0] - rm.origin[0]) / rm.res).astype(np.int64)
        h, w = rm.shape
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        nodes = -np.ones(len(path_xy), dtype=np.int64)
        nodes[ok] = self.node_of[r[ok], c[ok]]
        missing = np.nonzero(nodes < 0)[0]
        if len(missing) and self._tree is not None:  # e.g. start inside the inflation margin
            d, k = self._tree.query(path_xy[missing], distance_upper_bound=snap_radius_m)
            hit = np.isfinite(d)
            nodes[missing[hit]] = k[hit]
        return nodes, arc

    @staticmethod
    def lower_bound(path_xy, arc, obj_xy, r_max):
        """Lower bound on the detour to any viewpoint within r_max of obj_xy (straight lines)."""
        d = np.hypot(path_xy[:, 0] - obj_xy[0], path_xy[:, 1] - obj_xy[1])
        to_goal = math.hypot(obj_xy[0] - path_xy[-1, 0], obj_xy[1] - path_xy[-1, 1])
        return float(np.min(arc + d)) + to_goal - float(arc[-1]) - 2.0 * r_max

    def compute(self, path_xy, targets, params=None):
        """Detours for targets {object_id: (x, y, viewpoint_nodes[, max_m])} -> {object_id: DetourResult}.

        max_m is the object's own limit (e.g. the longest detour worth its value; default
        params.max_detour_m). Objects with max_m <= 0 or whose lower bound exceeds max_m are
        dropped before any search; results longer than max_m are left out.
        """
        params = params or DetourParams()
        path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
        if len(path_xy) < 2 or self.n == 0 or not targets:
            return {}
        nodes, arc = self._path_nodes(path_xy, params.snap_radius_m)
        keep = nodes >= 0
        if not keep[-1] or not keep.any():
            return {}  # goal is not in C-space on this grid
        cand = {}
        for oid, t in targets.items():
            x, y, vn = t[0], t[1], t[2]
            max_m = float(t[3]) if len(t) > 3 else params.max_detour_m
            if max_m > 0 and len(vn) and self.lower_bound(path_xy, arc, (x, y), params.r_max) <= max_m:
                cand[oid] = (x, y, vn, max_m)
        if not cand:
            return {}
        self.num_searches += 1

        # Virtual source n -> each path node, weight = arc length driven before leaving the path
        first_arc = {}
        first_idx = {}
        for i in np.nonzero(keep)[0]:
            nd = int(nodes[i])
            if nd not in first_arc:
                first_arc[nd], first_idx[nd] = float(arc[i]), int(i)
        src_nodes = np.fromiter(first_arc.keys(), dtype=np.int64)
        src_w = np.fromiter(first_arc.values(), dtype=float) + 1e-9  # explicit zeros are not edges
        n = self.n
        A = csr_matrix(
            (np.concatenate([self._wts, src_w]),
             (np.concatenate([self._rows, np.full(len(src_nodes), n)]), np.concatenate([self._cols, src_nodes]))),
            shape=(n + 1, n + 1),
        )
        limit = float(arc[-1]) + max(m for *_rest, m in cand.values()) + 1.0
        F, pred = dijkstra(A, directed=True, indices=n, limit=limit, return_predecessors=True)
        goal = int(nodes[-1])
        G = dijkstra(self.graph, directed=True, indices=goal, limit=limit)
        L = F[goal]
        if not np.isfinite(L):
            return {}

        out = {}
        for oid, (ox, oy, vn, max_m) in cand.items():
            total = F[vn] + G[vn] - L
            k = int(np.argmin(total))
            if not np.isfinite(total[k]) or total[k] > max_m:
                continue
            v = int(vn[k])
            route = [v]
            while pred[route[-1]] != n and pred[route[-1]] >= 0:
                route.append(int(pred[route[-1]]))
            route = route[::-1]  # path cell -> ... -> viewpoint
            # Driving the grid alongside the path ties with following it, so the source of the
            # chain can be far back: leave at the last route cell that is still "on the path"
            # (a path cell reached no cheaper than by following the path).
            leave_pos = 0
            for pos, nd in enumerate(route):
                if nd in first_arc and F[nd] >= first_arc[nd] - params.on_path_m:
                    leave_pos = pos
            leave = route[leave_pos]
            route_xy = self.xy[route[leave_pos:]]
            if len(route_xy) >= 2:
                d = route_xy[-1] - route_xy[max(len(route_xy) - 4, 0)]
                heading = math.atan2(d[1], d[0])
            else:
                i = first_idx[leave]
                d = path_xy[min(i + 1, len(path_xy) - 1)] - path_xy[max(i - 1, 0)]
                heading = math.atan2(d[1], d[0])
            detour = float(total[k])
            out[oid] = DetourResult(
                object_id=oid,
                detour_m=0.0 if detour <= params.on_path_m else detour,
                viewpoint_xy=(float(self.xy[v, 0]), float(self.xy[v, 1])),
                leave_index=first_idx[leave],
                leave_xy=(float(self.xy[leave, 0]), float(self.xy[leave, 1])),
                to_view_m=float(F[v] - F[leave]),
                arrival_heading=heading,
                route_xy=route_xy,
            )
        return out

    def shortest_path_xy(self, start_xy, goal_xy):
        """Grid shortest path start -> goal (world points), for offline tools and tests."""
        s = int(self._tree.query(start_xy)[1])
        g = int(self._tree.query(goal_xy)[1])
        _d, pred = dijkstra(self.graph, directed=True, indices=s, return_predecessors=True)
        if pred[g] < 0 and g != s:
            return None
        route = [g]
        while route[-1] != s:
            route.append(int(pred[route[-1]]))
        return self.xy[route[::-1]]
