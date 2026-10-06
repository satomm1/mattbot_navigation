"""Which roadmap edges an obstacle blocks.

Pure Python (numpy, scipy; no ROS). An edge is blocked when the robot can no longer get from one
of its nodes to the other inside the edge's corridor (the C-space free cells around its
polyline), with the obstacle inflated by the robot radius. A partial blockage that leaves room
to pass (e.g. a cart against one wall of a wide hallway) blocks nothing.

Only the static roadmap and this one obstacle are used: other obstacles never affect the result.
"""

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from matplotlib.path import Path
from scipy import ndimage

from navigation_utils.roadmap import disk_offsets

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class BlockingParams:
    blocking_search_radius: float = 2.0  # m; only edges this close to the obstacle are checked
    corridor_halfwidth_cap: float = 2.0  # m; corridor half-width = local free width, capped


def edge_key(u, v):
    return (u, v) if u <= v else (v, u)


# ---------- Footprints ----------


def footprint_cells(roadmap, coarse_cells=None, fine_cells=None, polygon=None, square=None):
    """(K, 2) unique coarse (row, col) cells covered by an obstacle, clipped to the map.

    coarse_cells: (K, 2) cells of the roadmap grid.
    fine_cells:   (K, 2) cells of the original occupancy grid (same origin).
    polygon:      (M, 2) world vertices; cells whose centre is inside, plus the centroid cell.
    square:       (x, y, width) world centre and side, drawn like dds_utils/belief.py does
                  (inclusive floor cell range, always at least the centre cell).
    """
    parts = []
    if coarse_cells is not None:
        parts.append(np.asarray(coarse_cells, dtype=int).reshape(-1, 2))
    if fine_cells is not None:
        parts.append(np.asarray(fine_cells, dtype=int).reshape(-1, 2) // roadmap.factor)
    if polygon is not None:
        parts.append(_polygon_cells(roadmap, np.asarray(polygon, dtype=float)))
    if square is not None:
        parts.append(_square_cells(roadmap, *square))
    if not parts:
        raise ValueError("footprint_cells: no footprint given")
    cells = np.unique(np.concatenate(parts, axis=0), axis=0)
    return cells[roadmap.in_bounds(cells)] if len(cells) else cells


def _square_cells(roadmap, x, y, width):
    half = width / 2.0
    ox, oy, res = roadmap.origin[0], roadmap.origin[1], roadmap.res
    i0, i1 = int(math.floor((x - half - ox) / res)), int(math.floor((x + half - ox) / res))
    j0, j1 = int(math.floor((y - half - oy) / res)), int(math.floor((y + half - oy) / res))
    rr, cc = np.mgrid[j0:j1 + 1, i0:i1 + 1]
    return np.stack([rr.ravel(), cc.ravel()], axis=1)


def _polygon_cells(roadmap, poly):
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    r0, c0 = roadmap.world_to_cell(lo[0], lo[1])
    r1, c1 = roadmap.world_to_cell(hi[0], hi[1])
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    rr, cc = rr.ravel(), cc.ravel()
    centres = np.stack(
        [roadmap.origin[0] + (cc + 0.5) * roadmap.res, roadmap.origin[1] + (rr + 0.5) * roadmap.res], axis=1
    )
    inside = Path(poly).contains_points(centres)
    cells = np.stack([rr[inside], cc[inside]], axis=1)
    centroid = roadmap.world_to_cell(*poly.mean(axis=0))
    return np.concatenate([cells, np.array([centroid])], axis=0)


def inflate_cells(roadmap, cells):
    """Footprint cells dilated by the robot radius (same disk as the C-space inflation)."""
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    if len(cells) == 0:
        return cells
    offs = disk_offsets(roadmap.params.robot_radius, roadmap.res)
    out = np.unique((cells[:, None, :] + offs[None, :, :]).reshape(-1, 2), axis=0)
    return out[roadmap.in_bounds(out)]


# ---------- Blocking ----------


def blocked_edges(roadmap, footprint, params=None):
    """frozenset of edges (u, v), u < v, blocked by an obstacle covering `footprint` cells.

    footprint: (K, 2) coarse cells of the obstacle (not inflated; see footprint_cells).

    An edge is blocked if there is no 8-connected free path inside its corridor between its two
    ends. When the inflated obstacle covers nodes, the robot cannot stand on them but may still
    pass around the obstacle. Covered nodes joined by edges form a cluster; each edge leaving the
    cluster is represented by its "arm", the first free cell of its polyline outside the obstacle.
    The cluster stays passable between the arms that remain connected around the obstacle (inside
    the union of the incident corridors): edges whose arm is cut off from the largest such group
    are blocked, and if no two arms connect, every edge at the cluster is blocked.
    """
    params = params or BlockingParams()
    inflated = inflate_cells(roadmap, footprint)
    if len(inflated) == 0:
        return frozenset()
    covered = set(map(tuple, inflated.tolist()))
    candidates = {edge_key(u, v) for u, v in roadmap.edges_near(inflated, params.blocking_search_radius)}
    covered_nodes = {n for e in candidates for n in e if tuple(roadmap.node_cell[n]) in covered}

    blocked = set()
    for cluster in nx.connected_components(roadmap.graph.subgraph(covered_nodes)):
        blocked |= _cut_edges_at_cluster(roadmap, cluster, inflated, params)
    for u, v in candidates:
        internal = u in covered_nodes and v in covered_nodes
        if (u, v) in blocked or internal:
            continue
        if edge_is_blocked(roadmap, u, v, inflated, covered, params):
            blocked.add((u, v))
    return frozenset(blocked)


def _poly_from(roadmap, u, v):
    """Polyline of edge (u, v) ordered from u to v."""
    poly = roadmap.graph.edges[u, v]["polyline"]
    return poly if u < v else poly[::-1]


def _halfwidth(roadmap, poly, params):
    return min(params.corridor_halfwidth_cap, float(roadmap.dist[poly[:, 0], poly[:, 1]].max()) + roadmap.res)


def _corridor(roadmap, polys, inflated, params):
    """Free corridor cells around the polylines, minus the inflated obstacle, on a cropped window.

    Returns (corridor mask, (r0, c0), touched) where touched says whether the obstacle removed
    any corridor cell.
    """
    hws = [_halfwidth(roadmap, p, params) for p in polys]
    pad = int(math.ceil(max(hws) / roadmap.res)) + 2
    allp = np.concatenate(polys, axis=0)
    h, w = roadmap.shape
    r0, r1 = max(int(allp[:, 0].min()) - pad, 0), min(int(allp[:, 0].max()) + pad + 1, h)
    c0, c1 = max(int(allp[:, 1].min()) - pad, 0), min(int(allp[:, 1].max()) + pad + 1, w)
    near = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for poly, hw in zip(polys, hws):
        on_poly = np.zeros_like(near)
        on_poly[poly[:, 0] - r0, poly[:, 1] - c0] = True
        near |= ndimage.distance_transform_edt(~on_poly) * roadmap.res <= hw + 1e-9
    corridor = roadmap.cfree[r0:r1, c0:c1] & near
    obst = inflated[(inflated[:, 0] >= r0) & (inflated[:, 0] < r1) & (inflated[:, 1] >= c0) & (inflated[:, 1] < c1)]
    touched = bool(len(obst)) and bool(corridor[obst[:, 0] - r0, obst[:, 1] - c0].any())
    if touched:
        corridor[obst[:, 0] - r0, obst[:, 1] - c0] = False
    return corridor, (r0, c0), touched


def _arm(poly_from_node, corridor, offset):
    """First polyline cell (walking away from the node) that is free in the corridor, or None."""
    r0, c0 = offset
    free = corridor[poly_from_node[:, 0] - r0, poly_from_node[:, 1] - c0]
    if not free.any():
        return None
    return tuple(poly_from_node[int(np.argmax(free))])


def _end_seed(roadmap, n, other, covered, corridor, offset):
    cell = tuple(roadmap.node_cell[n])
    if cell not in covered:
        return cell
    return _arm(_poly_from(roadmap, n, other), corridor, offset)


def _cut_edges_at_cluster(roadmap, cluster, inflated, params):
    """Edges at a cluster of covered nodes whose arm is cut off from the main group of arms."""
    g = roadmap.graph
    incident = {edge_key(n, m) for n in cluster for m in g.neighbors(n)}
    arms = [(n, m) for n in cluster for m in g.neighbors(n) if m not in cluster]
    if not arms:
        return incident
    polys = [g.edges[e]["polyline"] for e in incident]
    corridor, (r0, c0), _touched = _corridor(roadmap, polys, inflated, params)
    labels, _k = ndimage.label(corridor, structure=_EIGHT)
    groups = {}
    for n, m in arms:
        a = _arm(_poly_from(roadmap, n, m), corridor, (r0, c0))
        lab = 0 if a is None else int(labels[a[0] - r0, a[1] - c0])
        if lab:
            groups.setdefault(lab, []).append((n, m))
    main = max(groups.values(), key=len) if groups else []
    if len(main) < 2:
        return incident  # no passage through the covered area at all
    keep = {edge_key(n, m) for n, m in main}
    return {edge_key(n, m) for n, m in arms if edge_key(n, m) not in keep}


def edge_is_blocked(roadmap, u, v, inflated, covered, params):
    """Is there no 8-connected free path between the two ends of (u, v) inside its corridor?"""
    corridor, offset, touched = _corridor(roadmap, [_poly_from(roadmap, u, v)], inflated, params)
    if not touched:
        return False  # the obstacle does not reach this corridor
    su = _end_seed(roadmap, u, v, covered, corridor, offset)
    sv = _end_seed(roadmap, v, u, covered, corridor, offset)
    if su is None or sv is None:
        return True  # the whole polyline is covered
    labels, _k = ndimage.label(corridor, structure=_EIGHT)
    la = labels[su[0] - offset[0], su[1] - offset[1]]
    lb = labels[sv[0] - offset[0], sv[1] - offset[1]]
    return la == 0 or lb == 0 or la != lb
