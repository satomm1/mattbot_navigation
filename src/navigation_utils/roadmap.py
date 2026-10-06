"""Sparse topological roadmap of the free space, built once per static map.

Pure Python (numpy, scipy, scikit-image, networkx; no ROS). Grids use the ROS OccupancyGrid
layout: array[row, col] with row = y index and col = x index counted from the map origin.

Pipeline (see build_roadmap):
  1. Downsample the static map to roadmap_resolution; a coarse cell is occupied if any fine cell
     in it is occupied or unknown.
  2. C-space: obstacles inflated by robot_radius. Clearance is measured on the fine grid and
     sampled at each coarse cell centre (see _cspace for why).
  3. Skeletonize the C-space free region; junction (>= 3 skeleton neighbours) and endpoint
     (1 neighbour) pixels become nodes, branches between them become edges with their ordered
     pixel polylines. Long branches are split by corridor nodes every corridor_node_spacing m.
  4. Merge nodes closer than node_merge_radius (when the segment between them is free).
  5. Open areas wider than open_area_width get extra lattice nodes, joined by straight
     collision-free edges.
  6. Keep the largest connected component (warn about the others).

Edge weight = polyline arc length in metres (pixel steps of 1 or sqrt(2) cells), never the
straight-line distance between the two nodes.

The roadmap depends only on the static map and RoadmapParams. load_or_build_roadmap caches it on
disk keyed by a hash of both, so it is built once per map.
"""

import hashlib
import json
import logging
import math
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from navigation_utils.viewpoints import blocking_mask

log = logging.getLogger(__name__)

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = os.path.expanduser("~/.ros/mattbot_roadmap")

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class RoadmapParams:
    roadmap_resolution: float = 0.2  # m, coarse grid cell
    robot_radius: float = 0.4  # m, C-space inflation
    corridor_node_spacing: float = 1.5  # m between corridor nodes along a branch
    node_merge_radius: float = 0.3  # m
    open_area_width: float = 4.0  # m; wider free regions get lattice nodes
    open_area_node_spacing: float = 2.0  # m


def disk_offsets(radius_m, res):
    """(K, 2) int (drow, dcol) offsets of all cells whose centre is within radius_m of (0, 0)."""
    r = int(math.ceil(radius_m / res))
    dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
    keep = np.hypot(dr, dc) * res <= radius_m + 1e-9
    return np.stack([dr[keep], dc[keep]], axis=1)


def line_cells(c0, c1):
    """(K, 2) int cells of the 8-connected digital line from cell c0 to c1 (both included)."""
    (r0, k0), (r1, k1) = c0, c1
    n = max(abs(r1 - r0), abs(k1 - k0)) + 1
    rows = np.rint(np.linspace(r0, r1, n)).astype(int)
    cols = np.rint(np.linspace(k0, k1, n)).astype(int)
    return np.stack([rows, cols], axis=1)


def densify(cells):
    """Fill gaps between consecutive cells of a polyline so it is 8-connected."""
    cells = np.asarray(cells, dtype=int)
    if len(cells) < 2:
        return cells
    out = [cells[:1]]
    for a, b in zip(cells[:-1], cells[1:]):
        if np.abs(b - a).max() <= 1:
            if (b != a).any():
                out.append(b[None])
        else:
            out.append(line_cells(tuple(a), tuple(b))[1:])
    return np.concatenate(out, axis=0)


def arc_length_cells(cells):
    """Arc length of a cell polyline, in cells (sum of step lengths)."""
    cells = np.asarray(cells, dtype=float)
    if len(cells) < 2:
        return 0.0
    return float(np.hypot(*np.diff(cells, axis=0).T).sum())


def normalize_resolution(resolution):
    """OccupancyGrid.info.resolution is float32 (0.05 -> 0.0500000007); undo that so the same map
    gives the same roadmap (and cache entry) from a ROS message and from a JSON/YAML file."""
    return round(float(resolution), 6)


def map_fingerprint(occupancy_data, width, height, resolution, origin_xy, params):
    """SHA-256 of the static map (cells + geometry) and the roadmap parameters."""
    resolution = normalize_resolution(resolution)
    origin_xy = (round(float(origin_xy[0]), 6), round(float(origin_xy[1]), 6))
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(occupancy_data, dtype=np.int8)).tobytes())
    meta = {
        "version": CACHE_VERSION,
        "width": int(width),
        "height": int(height),
        "resolution": round(float(resolution), 9),
        "origin": [round(float(origin_xy[0]), 9), round(float(origin_xy[1]), 9)],
        "params": asdict(params),
    }
    h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()


class Roadmap:
    """Topological roadmap: graph + the coarse C-space grid it was built on.

    graph: networkx.Graph with integer nodes 0..N-1. Node attrs: xy (world, m), cell (row, col),
    kind ("junction", "endpoint", "corridor", "loop", "open"). Edge attrs: weight (m) and polyline
    ((K, 2) int coarse cells, ordered from the smaller node id to the larger).
    """

    def __init__(self, graph, occ, cfree, dist, res, origin, fine_resolution, factor, params, fingerprint):
        self.graph = graph
        self.occ = occ  # coarse occupied (incl. unknown)
        self.cfree = cfree  # coarse C-space free
        self.dist = dist  # m, distance from each C-space free cell to the C-space boundary
        self.res = res
        self.origin = origin
        self.fine_resolution = fine_resolution
        self.factor = factor  # fine cells per coarse cell (per axis)
        self.params = params
        self.fingerprint = fingerprint
        self._index()

    def _index(self):
        n = self.graph.number_of_nodes()
        self.node_xy = np.array([self.graph.nodes[i]["xy"] for i in range(n)], dtype=float).reshape(n, 2)
        self.node_cell = np.array([self.graph.nodes[i]["cell"] for i in range(n)], dtype=int).reshape(n, 2)
        self.edge_list = [tuple(sorted(e)) for e in self.graph.edges()]
        self._node_tree = cKDTree(self.node_xy) if n else None
        if self.edge_list:
            polys = [self.graph.edges[e]["polyline"] for e in self.edge_list]
            self._poly_cells = np.concatenate(polys, axis=0)
            self._poly_edge = np.concatenate([np.full(len(p), k) for k, p in enumerate(polys)])
            self._poly_tree = cKDTree(self._poly_cells)
        else:
            self._poly_cells = np.zeros((0, 2), dtype=int)
            self._poly_edge = np.zeros(0, dtype=int)
            self._poly_tree = None

    @property
    def shape(self):
        return self.cfree.shape

    def world_to_cell(self, x, y):
        return (
            int(math.floor((y - self.origin[1]) / self.res)),
            int(math.floor((x - self.origin[0]) / self.res)),
        )

    def cell_to_world(self, row, col):
        return (self.origin[0] + (col + 0.5) * self.res, self.origin[1] + (row + 0.5) * self.res)

    def in_bounds(self, cells):
        cells = np.atleast_2d(cells)
        h, w = self.shape
        return (cells[:, 0] >= 0) & (cells[:, 0] < h) & (cells[:, 1] >= 0) & (cells[:, 1] < w)

    def segment_free(self, c0, c1, allow_start_occupied=False):
        """Is the straight line c0 -> c1 inside C-space free (and on the map)?

        allow_start_occupied: the start may be inside the inflation margin (e.g. a landmark next
        to a wall): a leading run of non-C-space cells is allowed as long as none of them is an
        actual obstacle.
        """
        cells = line_cells(c0, c1)
        ok = self.in_bounds(cells)
        if not ok.all():
            return False
        free = self.cfree[cells[:, 0], cells[:, 1]]
        if allow_start_occupied:
            first = int(np.argmax(free)) if free.any() else len(free)
            if first == len(free) or self.occ[cells[:first, 0], cells[:first, 1]].any():
                return False
            free = free[first:]
        return bool(free.all())

    def nearest_node(self, x, y, k=16):
        """Nearest node reachable from (x, y) along a straight collision-free line.

        Falls back to the Euclidean nearest node (with a warning) if none of the k nearest is
        visible.
        """
        if self._node_tree is None:
            raise ValueError("empty roadmap")
        k = min(k, len(self.node_xy))
        _d, idx = self._node_tree.query([x, y], k=k)
        idx = np.atleast_1d(idx)
        c0 = self.world_to_cell(x, y)
        for i in idx:
            if self.segment_free(c0, tuple(self.node_cell[i]), allow_start_occupied=True):
                return int(i)
        log.warning("roadmap: no node visible from (%.2f, %.2f); using the nearest one", x, y)
        return int(idx[0])

    def edges_near(self, cells, dist_m):
        """Set of edges (u, v), u < v, whose polylines come within dist_m of any of cells (K, 2)."""
        cells = np.atleast_2d(np.asarray(cells))
        if self._poly_tree is None or cells.size == 0:
            return set()
        hits = self._poly_tree.query_ball_point(cells, dist_m / self.res + 1e-9)
        found = set()
        for lst in hits:
            found.update(self._poly_edge[lst].tolist())
        return {self.edge_list[k] for k in found}

    def edge_weight(self, u, v):
        return self.graph.edges[u, v]["weight"]


# ---------- Build ----------


def coarse_occupancy(blocked, factor):
    """Block-reduce a fine (H, W) occupied mask: coarse cell occupied if any fine cell is."""
    h, w = blocked.shape
    hc, wc = -(-h // factor), -(-w // factor)
    padded = np.ones((hc * factor, wc * factor), dtype=bool)  # off-map counts as occupied
    padded[:h, :w] = blocked
    return padded.reshape(hc, factor, wc, factor).any(axis=(1, 3))


def _cspace(blocked_fine, fine_res, factor, robot_radius):
    """Coarse C-space free mask, distance (m) to the C-space boundary, and clearance (m).

    Clearance is measured exactly on the fine grid (distance from each fine cell to the nearest
    occupied fine cell) and sampled at the centre fine cell of each coarse cell. Inflating the
    coarse grid instead (coarse cell centre to coarse cell centre) loses up to a coarse cell per
    wall, which closes ~1 m doorways at 0.2 m; sampling keeps the fine-grid connectivity.
    A coarse cell with clearance > robot_radius never contains an occupied fine cell, so
    C-space free is always inside the coarse free cells.
    """
    clear_fine = ndimage.distance_transform_edt(~blocked_fine) * fine_res
    h, w = blocked_fine.shape
    hc, wc = -(-h // factor), -(-w // factor)
    rows = np.minimum(np.arange(hc) * factor + factor // 2, h - 1)
    cols = np.minimum(np.arange(wc) * factor + factor // 2, w - 1)
    clearance = clear_fine[np.ix_(rows, cols)]
    cfree = clearance > robot_radius + 1e-9
    cfree[0, :] = cfree[-1, :] = cfree[:, 0] = cfree[:, -1] = False  # keep tracing inside the grid
    dist = ndimage.distance_transform_edt(cfree) * fine_res * factor
    return cfree, dist, clearance, clear_fine > robot_radius + 1e-9


def _skeleton_graph(skel):
    """Trace a skeleton into nodes (cells) and branches (node_a, node_b, cell polyline)."""
    h, w = skel.shape
    deg = ndimage.convolve(skel.astype(np.int16), _EIGHT.astype(np.int16), mode="constant") - 1
    deg[~skel] = 0

    node_of = -np.ones(skel.shape, dtype=int)  # node id per pixel (junction clusters share one)
    nodes = []  # (cell, kind)

    junction = skel & (deg >= 3)
    labels, n_j = ndimage.label(junction, structure=_EIGHT)
    if n_j:
        for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
            rr, cc = np.nonzero(labels[sl] == lab)
            rr, cc = rr + sl[0].start, cc + sl[1].start
            centre = np.array([rr.mean(), cc.mean()])
            k = int(np.argmin((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2))
            node_of[rr, cc] = len(nodes)
            nodes.append(((int(rr[k]), int(cc[k])), "junction"))
    for r, c in zip(*np.nonzero(skel & (deg <= 1))):
        node_of[r, c] = len(nodes)
        nodes.append(((int(r), int(c)), "endpoint"))

    visited = np.zeros(skel.shape, dtype=bool)
    branches = []
    seen_direct = set()

    def neighbours(r, c):
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                yield rr, cc

    def walk(start_node, start_pixel, first):
        """Follow non-node pixels from `first` (adjacent to start_pixel) to the next node."""
        path = [nodes[start_node][0], start_pixel, first]
        prev, cur = start_pixel, first
        while node_of[cur] < 0:
            visited[cur] = True
            nxt = None
            for p in neighbours(*cur):
                if p == prev or (node_of[p] < 0 and visited[p]):
                    continue
                if node_of[p] >= 0 and node_of[p] == start_node and len(path) <= 3:
                    continue  # do not step straight back into the start cluster
                nxt = p
                if node_of[p] >= 0:
                    break  # prefer arriving at a node
            if nxt is None:
                return None  # dead end without a node (should not happen on a clean skeleton)
            prev, cur = cur, nxt
            path.append(cur)
        end = int(node_of[cur])
        path.append(nodes[end][0])
        return end, path

    def trace_from(nid):
        members = np.argwhere(node_of == nid) if nodes[nid][1] == "junction" else [nodes[nid][0]]
        for pr, pc in members:
            pr, pc = int(pr), int(pc)
            for q in neighbours(pr, pc):
                if node_of[q] == nid:
                    continue
                if node_of[q] >= 0:  # two nodes touching directly
                    key = tuple(sorted((nid, int(node_of[q]))))
                    if key not in seen_direct:
                        seen_direct.add(key)
                        branches.append((key[0], key[1], [nodes[key[0]][0], nodes[key[1]][0]]))
                    continue
                if visited[q]:
                    continue
                res = walk(nid, (pr, pc), q)
                if res is not None:
                    branches.append((nid, res[0], res[1]))

    for nid in range(len(nodes)):
        trace_from(nid)

    # Closed loops with no junction or endpoint: seed a node on each and trace it.
    rest = skel & ~visited & (node_of < 0)
    while rest.any():
        r, c = map(int, np.argwhere(rest)[0])
        node_of[r, c] = len(nodes)
        nodes.append(((r, c), "loop"))
        trace_from(len(nodes) - 1)
        rest = skel & ~visited & (node_of < 0)

    return nodes, branches


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class _Builder:
    """Accumulates nodes (cells) and edges (polylines) before the final networkx graph."""

    def __init__(self, cfree, res):
        self.cfree = cfree
        self.res = res
        self.cells = []
        self.kinds = []
        self.edges = {}  # (u, v) u < v -> polyline (cells from u to v)

    def add_node(self, cell, kind):
        self.cells.append((int(cell[0]), int(cell[1])))
        self.kinds.append(kind)
        return len(self.cells) - 1

    def add_edge(self, u, v, poly):
        """Add (keep the shorter of parallel edges); returns False if a parallel edge existed."""
        if u == v:
            return True
        poly = densify(poly)
        if u > v:
            u, v, poly = v, u, poly[::-1]
        old = self.edges.get((u, v))
        if old is not None:
            if arc_length_cells(poly) < arc_length_cells(old):
                self.edges[(u, v)] = poly
            return False
        self.edges[(u, v)] = poly
        return True

    def add_branch(self, a, b, poly, spacing_cells):
        """Add a skeleton branch, split by corridor nodes about every spacing_cells."""
        poly = densify(poly)
        steps = np.hypot(*np.diff(poly, axis=0).T) if len(poly) > 1 else np.zeros(0)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        length = s[-1]
        n_seg = max(1, int(round(length / spacing_cells)))
        if a == b:
            n_seg = max(n_seg, 3)  # a loop back to the same node needs >= 2 interior nodes
        elif (min(a, b), max(a, b)) in self.edges:
            n_seg = max(n_seg, 2)  # parallel branch: keep it distinct with a midpoint node
        if n_seg == 1:
            self.add_edge(a, b, poly)
            return
        cut_idx = [int(np.argmin(np.abs(s - length * k / n_seg))) for k in range(1, n_seg)]
        cut_idx = sorted({i for i in cut_idx if 0 < i < len(poly) - 1})
        prev_node, prev_i = a, 0
        for i in cut_idx:
            nid = self.add_node(poly[i], "corridor")
            self.add_edge(prev_node, nid, poly[prev_i:i + 1])
            prev_node, prev_i = nid, i
        self.add_edge(prev_node, b, poly[prev_i:])


def _merge_close_nodes(b, merge_cells, segment_free):
    cells = np.array(b.cells, dtype=float)
    if len(cells) < 2:
        return b
    uf = _UnionFind(len(cells))
    for i, j in cKDTree(cells).query_pairs(merge_cells):
        if segment_free(b.cells[i], b.cells[j]):
            uf.union(i, j)
    groups = {}
    for i in range(len(cells)):
        groups.setdefault(uf.find(i), []).append(i)
    if all(len(g) == 1 for g in groups.values()):
        return b

    out = _Builder(b.cfree, b.res)
    new_id = {}
    rank = {"junction": 0, "endpoint": 1, "loop": 2, "corridor": 3, "open": 4}
    for root, members in groups.items():
        centre = cells[members].mean(axis=0)
        rep = min(members, key=lambda m: np.sum((cells[m] - centre) ** 2))
        kind = min((b.kinds[m] for m in members), key=lambda k: rank[k])
        nid = out.add_node(b.cells[rep], kind)
        for m in members:
            new_id[m] = nid
    for (u, v), poly in b.edges.items():
        nu, nv = new_id[u], new_id[v]
        if nu == nv:
            continue
        poly = np.concatenate([np.array([out.cells[nu]]), poly, np.array([out.cells[nv]])])
        out.add_edge(nu, nv, poly)
    return out


def _add_open_area_nodes(b, cfree, clearance, params, res, segment_free):
    """Lattice nodes in regions wider than open_area_width, joined by straight free segments.

    The open region is the union of all free disks of diameter open_area_width (a morphological
    opening of the free space), restricted to C-space free.
    """
    half = params.open_area_width / 2.0
    core = 2.0 * clearance > params.open_area_width  # centres of such disks
    if not core.any():
        return
    from_core = ndimage.distance_transform_edt(~core) * res
    open_mask = cfree & (from_core <= half - params.robot_radius + 1e-9)
    step = max(1, int(round(params.open_area_node_spacing / res)))
    h, w = cfree.shape
    existing = cKDTree(np.array(b.cells, dtype=float)) if b.cells else None
    merge_cells = params.node_merge_radius / res
    lattice = {}
    for r in range(step // 2, h, step):
        for c in range(step // 2, w, step):
            if not open_mask[r, c]:
                continue
            if existing is not None and existing.query([r, c])[0] <= merge_cells:
                continue
            lattice[(r // step, c // step)] = b.add_node((r, c), "open")
    if not lattice:
        return
    base_n = len(b.cells) - len(lattice)
    for (gi, gj), nid in lattice.items():
        for di, dj in [(0, 1), (1, -1), (1, 0), (1, 1)]:  # each lattice pair once
            other = lattice.get((gi + di, gj + dj))
            if other is not None and segment_free(b.cells[nid], b.cells[other]):
                b.add_edge(nid, other, line_cells(b.cells[nid], b.cells[other]))
    if base_n:
        base = cKDTree(np.array(b.cells[:base_n], dtype=float))
        radius = step * math.sqrt(2.0) + 1e-6
        for nid in lattice.values():
            for j in base.query_ball_point(b.cells[nid], radius):
                if segment_free(b.cells[nid], b.cells[j]):
                    b.add_edge(j, nid, line_cells(b.cells[j], b.cells[nid]))


def _check_coarse_connectivity(fine_cfree, fine_res, factor, cfree, params, origin):
    """Warn where coarsening split a free region that is connected at full resolution."""
    fine_lab, n_fine = ndimage.label(fine_cfree, structure=_EIGHT)
    coarse_lab, _n = ndimage.label(cfree, structure=_EIGHT)
    if n_fine == 0:
        return []
    rr, cc = np.nonzero(fine_lab)
    cr, ccol = rr // factor, cc // factor
    ok = (cr < cfree.shape[0]) & (ccol < cfree.shape[1])
    rr, cc, cr, ccol = rr[ok], cc[ok], cr[ok], ccol[ok]
    cl = coarse_lab[cr, ccol]
    keep = cl > 0
    pairs = np.unique(np.stack([fine_lab[rr[keep], cc[keep]], cl[keep]], axis=1), axis=0)
    splits = []
    fine_ids, counts = np.unique(pairs[:, 0], return_counts=True)
    for fid in fine_ids[counts > 1]:
        coarse_ids = pairs[pairs[:, 0] == fid, 1]
        sizes = ndimage.sum(np.ones_like(coarse_lab), coarse_lab, coarse_ids)
        if np.sort(sizes)[-2] * (params.roadmap_resolution ** 2) < 1.0:
            continue  # the split-off piece is tiny (< 1 m^2): not a real passage
        rows, cols = np.nonzero(fine_lab == fid)
        x = origin[0] + (cols.mean() + 0.5) * fine_res
        y = origin[1] + (rows.mean() + 0.5) * fine_res
        splits.append((int(fid), len(coarse_ids), x, y))
        log.warning(
            "roadmap: a free region around (%.1f, %.1f) is connected at %.2f m but split into %d parts "
            "at %.2f m (narrow passage lost); consider a smaller roadmap robot_radius",
            x, y, fine_res, len(coarse_ids), params.roadmap_resolution,
        )
    return splits


def build_roadmap(occupancy_data, width, height, resolution, origin_xy, params=None, fingerprint=None):
    """Build the roadmap from a ROS OccupancyGrid (data row-major, -1 unknown, >= 50 occupied)."""
    params = params or RoadmapParams()
    resolution = normalize_resolution(resolution)
    origin = (round(float(origin_xy[0]), 6), round(float(origin_xy[1]), 6))  # float32 in ROS messages too
    blocked_fine = blocking_mask(occupancy_data, width, height)
    factor = max(1, int(round(params.roadmap_resolution / resolution)))
    res = factor * resolution
    occ = coarse_occupancy(blocked_fine, factor)
    cfree, dist, clearance, fine_cfree = _cspace(blocked_fine, resolution, factor, params.robot_radius)

    def segment_free(c0, c1):
        cells = line_cells(c0, c1)
        return bool(cfree[cells[:, 0], cells[:, 1]].all())

    skel = skeletonize(cfree)
    skel_nodes, branches = _skeleton_graph(skel)

    b = _Builder(cfree, res)
    for cell, kind in skel_nodes:
        b.add_node(cell, kind)
    spacing_cells = params.corridor_node_spacing / res
    for a, z, poly in branches:
        b.add_branch(a, z, poly, spacing_cells)

    b = _merge_close_nodes(b, params.node_merge_radius / res, segment_free)
    _add_open_area_nodes(b, cfree, clearance, params, res, segment_free)

    g = nx.Graph()
    for i, (cell, kind) in enumerate(zip(b.cells, b.kinds)):
        g.add_node(i, cell=cell, kind=kind, xy=(origin[0] + (cell[1] + 0.5) * res, origin[1] + (cell[0] + 0.5) * res))
    for (u, v), poly in b.edges.items():
        g.add_edge(u, v, weight=arc_length_cells(poly) * res, polyline=np.asarray(poly, dtype=np.int32))

    g = _largest_component(g)
    _check_coarse_connectivity(fine_cfree, resolution, factor, cfree, params, origin)
    if fingerprint is None:
        fingerprint = map_fingerprint(occupancy_data, width, height, resolution, origin, params)
    return Roadmap(g, occ, cfree, dist, res, origin, resolution, factor, params, fingerprint)


def _largest_component(g):
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    if not comps:
        return g
    for comp in comps[1:]:
        xy = np.array([g.nodes[n]["xy"] for n in comp])
        log.warning(
            "roadmap: dropping disconnected component of %d node(s) around (%.1f, %.1f)",
            len(comp), xy[:, 0].mean(), xy[:, 1].mean(),
        )
    keep = g.subgraph(comps[0]).copy()
    return nx.convert_node_labels_to_integers(keep, ordering="sorted")


# ---------- Disk cache ----------


def _atomic_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def roadmap_cache_path(cache_dir, fingerprint):
    return os.path.join(cache_dir, "roadmap_%s.pkl" % fingerprint[:16])


def load_or_build_roadmap(occupancy_data, width, height, resolution, origin_xy, params=None, cache_dir=None):
    """Load the roadmap for this map + params from cache_dir, or build and save it.

    Returns (roadmap, loaded_from_cache).
    """
    params = params or RoadmapParams()
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    fp = map_fingerprint(occupancy_data, width, height, resolution, origin_xy, params)
    path = roadmap_cache_path(cache_dir, fp)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("version") == CACHE_VERSION and blob.get("fingerprint") == fp:
                return blob["roadmap"], True
            log.warning("roadmap: cache %s is stale; rebuilding", path)
        except Exception as e:  # corrupt or incompatible cache: rebuild
            log.warning("roadmap: could not read cache %s (%s); rebuilding", path, e)
    roadmap = build_roadmap(occupancy_data, width, height, resolution, origin_xy, params, fingerprint=fp)
    try:
        _atomic_pickle({"version": CACHE_VERSION, "fingerprint": fp, "roadmap": roadmap}, path)
    except OSError as e:
        log.warning("roadmap: could not write cache %s (%s)", path, e)
    return roadmap, False
