"""Driving distance for a scout: an idle robot that leaves only to observe an object.

Pure Python (numpy, scipy; no ROS). Shares the detour planner's C-space grid
(navigation_utils/detour.py, roadmap.cfree at 0.2 m, 8-connected); nothing is rebuilt.

A scout starts at the robot's position s with no goal of its own. For an object with viewpoints V
(its viewshed mapped to C-space cells, DetourPlanner.viewpoint_nodes):

    to_view_m = min_{v in V} d(s, v)       one Dijkstra from s for all objects together
    return_m  = d(v*, s) = to_view_m       the grid is symmetric

Both legs are reported; which of them a scout pays is the cost model's choice
(observation_policy.ScoutCostModel). Far objects are skipped before any search with
    to_view_m >= |s - o| - r_max - res / sqrt(2)
(s is the start cell centre; every viewpoint cell centre lies within r_max of the object o, plus
half a cell diagonal because a coarse cell is a viewpoint if any fine viewshed cell lies in it).
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra


@dataclass
class ScoutParams:
    max_scout_m: float = 30.0  # objects whose viewpoints are further away (one way) are ignored
    r_max: float = 3.5  # m; viewshed radius (for the far-object lower bound)
    snap_radius_m: float = 0.5  # a start off C-space snaps to a free cell this close


@dataclass
class ScoutResult:
    object_id: str
    viewpoint_xy: Tuple[float, float]  # where to look from (coarse cell centre)
    start_xy: Tuple[float, float]  # the robot's C-space start cell
    to_view_m: float  # start -> viewpoint
    return_m: float  # viewpoint -> start
    arrival_heading: float  # heading when reaching the viewpoint (rad)
    route_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # start -> viewpoint

    def to_dict(self):
        return {
            "viewpoint": list(self.viewpoint_xy),
            "start": list(self.start_xy),
            "to_view_m": self.to_view_m,
            "return_m": self.return_m,
            "arrival_heading": self.arrival_heading,
        }


class ScoutPlanner:
    """Scout distances on a DetourPlanner's grid graph (built once per roadmap)."""

    def __init__(self, detour_planner):
        self.grid = detour_planner
        self.num_searches = 0  # queries that ran the Dijkstra search (diagnostics / tests)

    def viewpoint_nodes(self, viewshed):
        return self.grid.viewpoint_nodes(viewshed)

    @staticmethod
    def lower_bound(start_xy, obj_xy, r_max, slack_m=0.0):
        """Lower bound on the distance from start_xy to any viewpoint within r_max + slack_m of obj_xy."""
        return math.hypot(obj_xy[0] - start_xy[0], obj_xy[1] - start_xy[1]) - r_max - slack_m

    def compute(self, start_xy, targets, params=None, start_heading=0.0):
        """Scouts for targets {object_id: (x, y, viewpoint_nodes[, max_m])} -> {object_id: ScoutResult}.

        max_m is the object's own limit on to_view_m (default params.max_scout_m). Objects with
        max_m <= 0 or whose lower bound exceeds max_m are dropped before any search; results
        further than max_m are left out. start_heading is the arrival heading of a scout that
        is already at a viewpoint.
        """
        params = params or ScoutParams()
        g = self.grid
        if g.n == 0 or not targets:
            return {}
        start = int(g.snap_nodes([start_xy], params.snap_radius_m)[0])
        if start < 0:
            return {}  # robot is not in C-space on this grid
        sxy = (float(g.xy[start, 0]), float(g.xy[start, 1]))
        slack = g.roadmap.res / math.sqrt(2.0)
        cand = {}
        for oid, t in targets.items():
            x, y, vn = t[0], t[1], t[2]
            max_m = min(float(t[3]) if len(t) > 3 else params.max_scout_m, params.max_scout_m)
            if max_m > 0 and len(vn) and self.lower_bound(sxy, (x, y), params.r_max, slack) <= max_m:
                cand[oid] = (x, y, vn, max_m)
        if not cand:
            return {}
        self.num_searches += 1

        limit = max(m for *_rest, m in cand.values()) + 1.0
        F, pred = dijkstra(g.graph, directed=True, indices=start, limit=limit, return_predecessors=True)

        out = {}
        for oid, (_ox, _oy, vn, max_m) in cand.items():
            dist = F[vn]
            k = int(np.argmin(dist))
            if not np.isfinite(dist[k]) or dist[k] > max_m:
                continue
            v = int(vn[k])
            route = [v]
            while pred[route[-1]] >= 0:
                route.append(int(pred[route[-1]]))
            route_xy = g.xy[route[::-1]]  # start -> viewpoint
            if len(route_xy) >= 2:
                d = route_xy[-1] - route_xy[max(len(route_xy) - 4, 0)]
                heading = math.atan2(d[1], d[0])
            else:
                heading = float(start_heading)
            out[oid] = ScoutResult(
                object_id=oid,
                viewpoint_xy=(float(g.xy[v, 0]), float(g.xy[v, 1])),
                start_xy=sxy,
                to_view_m=float(dist[k]),
                return_m=float(dist[k]),
                arrival_heading=heading,
                route_xy=route_xy,
            )
        return out
