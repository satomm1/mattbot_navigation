"""Obstacle importance I_o: expected extra travel distance per trip caused by one obstacle.

Pure Python (numpy, scipy, networkx; no ROS). For a trip set T = {(s_k, t_k)} with weights w_k
(sum 1) on the roadmap:

    delta_k = min(d_blocked(s_k, t_k) - d_free(s_k, t_k), d_max)   (d_max if t_k is cut off)
    I_o = sum_k w_k * delta_k                                        (m per trip)
    frac_affected = sum_k w_k * [delta_k > 0];  mean_detour_if_affected = I_o / frac_affected

d_blocked removes only this obstacle's blocked edges (edge_blocking.blocked_edges) from the
obstacle-free roadmap. I_o is a property of one obstacle against the static map: other obstacles
are never considered, so adding or removing an obstacle never changes another obstacle's I_o.
(Two obstacles that only together cut off an area each get their individual, possibly low, I_o.)

Discovery cost I_o_disc: the robot believes the route is clear, drives the free shortest path,
finds the obstacle at the first blocked edge (its endpoint x nearer to s) and replans from x:
    delta_disc_k = min(d_free(s, x) + d_blocked(x, t) - d_free(s, t), d_max) >= delta_k.

Trip sets (TripSet):
  uniform_landmarks:  all ordered pairs of landmarks, equal weights. Landmarks come from a YAML
                      file of named local-map coordinates, or num_random_landmarks random nodes.
  weighted_landmarks: the same pairs weighted by counts from a CSV (start, goal, count).
  TripSet.from_pairs: any explicit (start xy, goal xy) pairs.
"""

import csv
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from navigation_utils.edge_blocking import BlockingParams, blocked_edges, footprint_cells

log = logging.getLogger(__name__)

UNIFORM_LANDMARKS = "uniform_landmarks"
WEIGHTED_LANDMARKS = "weighted_landmarks"
EXPLICIT = "explicit"
TRIP_MODES = (UNIFORM_LANDMARKS, WEIGHTED_LANDMARKS)

DIST_CACHE_VERSION = 1


@dataclass
class ImportanceParams:
    d_max: float = 50.0  # m; cap on one trip's extra distance (and cost of an unreachable goal)
    eps_path_rel: float = 1e-6  # relative tolerance for "edge lies on a shortest path"
    trip_mode: str = UNIFORM_LANDMARKS
    landmarks_file: Optional[str] = None  # YAML {name: [x, y]} in the local map frame
    trip_counts_file: Optional[str] = None  # CSV start_landmark, goal_landmark, count
    num_random_landmarks: int = 50
    seed: int = 0


@dataclass
class ImportanceResult:
    I_o: float
    I_o_disc: float
    frac_affected: float
    mean_detour_if_affected: float
    num_trips: int
    num_affected: int
    blocked_edges: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    compute_time_s: float = 0.0

    def to_dict(self):
        return {
            "I_o": self.I_o,
            "I_o_disc": self.I_o_disc,
            "frac_affected": self.frac_affected,
            "mean_detour_if_affected": self.mean_detour_if_affected,
            "num_trips": self.num_trips,
            "num_affected": self.num_affected,
            "blocked_edges": sorted([list(e) for e in self.blocked_edges]),
            "compute_time_s": self.compute_time_s,
        }


# ---------- Trip sets ----------


@dataclass
class TripSet:
    mode: str
    landmark_nodes: np.ndarray  # (L,) roadmap node per landmark
    landmark_names: list  # (L,) names (None for random landmarks)
    sources: np.ndarray  # (T,) landmark index of each trip's start
    targets: np.ndarray  # (T,) landmark index of each trip's goal
    weights: np.ndarray  # (T,) sum 1

    @property
    def key(self):
        h = hashlib.sha256()
        h.update(self.mode.encode())
        for a in (self.landmark_nodes, self.sources, self.targets):
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(self.weights, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]

    def __len__(self):
        return len(self.weights)

    @classmethod
    def from_pairs(cls, roadmap, pairs_xy, weights=None, mode=EXPLICIT):
        """Explicit trips: pairs_xy = [((sx, sy), (tx, ty)), ...] in the local map frame."""
        nodes, src, dst = [], [], []
        index = {}

        def lm(xy):
            n = roadmap.nearest_node(*xy)
            if n not in index:
                index[n] = len(nodes)
                nodes.append(n)
            return index[n]

        for s_xy, t_xy in pairs_xy:
            src.append(lm(s_xy))
            dst.append(lm(t_xy))
        w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
        return cls(mode, np.array(nodes, dtype=int), [None] * len(nodes), np.array(src, dtype=int),
                   np.array(dst, dtype=int), _normalize(w))


def _normalize(w):
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("trip weights sum to zero")
    return w / total


def load_landmarks(path):
    """YAML {name: [x, y], ...} (local map frame) -> list of (name, x, y)."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and "landmarks" in data:
        data = data["landmarks"]
    out = []
    for name, xy in data.items():
        out.append((str(name), float(xy[0]), float(xy[1])))
    if len(out) < 2:
        raise ValueError("%s: need at least 2 landmarks" % path)
    return out


def load_trip_counts(path):
    """CSV rows start_landmark, goal_landmark, count (optional header) -> list of (s, t, count)."""
    rows = []
    with open(path, newline="") as f:
        for rec in csv.reader(f):
            if not rec or rec[0].strip().startswith("#"):
                continue
            try:
                rows.append((rec[0].strip(), rec[1].strip(), float(rec[2])))
            except (IndexError, ValueError):
                if rows:  # anything but a header line is an error
                    raise ValueError("%s: bad row %r" % (path, rec))
    return rows


# TODO: load trip frequencies from logged fleet goal history (mattbot_database goals table),
# e.g. by snapping consecutive goals of each robot to landmarks and counting the pairs.


def build_trip_set(roadmap, params):
    """Trip set for params.trip_mode on this roadmap."""
    if params.trip_mode not in TRIP_MODES:
        raise ValueError("unknown trip_mode %r (expected one of %s)" % (params.trip_mode, ", ".join(TRIP_MODES)))
    if params.landmarks_file:
        named = load_landmarks(params.landmarks_file)
        names = [n for n, _x, _y in named]
        nodes = np.array([roadmap.nearest_node(x, y) for _n, x, y in named], dtype=int)
    else:
        if params.trip_mode == WEIGHTED_LANDMARKS:
            raise ValueError("weighted_landmarks needs a landmarks_file (the CSV refers to landmark names)")
        n = roadmap.graph.number_of_nodes()
        rng = np.random.default_rng(params.seed)
        nodes = np.sort(rng.choice(n, size=min(params.num_random_landmarks, n), replace=False))
        names = [None] * len(nodes)

    L = len(nodes)
    if params.trip_mode == UNIFORM_LANDMARKS:
        ss, tt = np.nonzero(~np.eye(L, dtype=bool))
        return TripSet(UNIFORM_LANDMARKS, nodes, names, ss, tt, np.full(len(ss), 1.0 / max(len(ss), 1)))

    if not params.trip_counts_file:
        raise ValueError("weighted_landmarks needs a trip_counts_file")
    index = {name: i for i, name in enumerate(names)}
    counts = {}
    for s, t, c in load_trip_counts(params.trip_counts_file):
        if s not in index or t not in index:
            log.warning("trip counts: unknown landmark in (%s, %s); skipped", s, t)
            continue
        if s == t or c <= 0:
            continue
        counts[(index[s], index[t])] = counts.get((index[s], index[t]), 0.0) + c
    if not counts:
        raise ValueError("%s: no usable trips" % params.trip_counts_file)
    pairs = sorted(counts)
    return TripSet(
        WEIGHTED_LANDMARKS, nodes, names,
        np.array([p[0] for p in pairs], dtype=int), np.array([p[1] for p in pairs], dtype=int),
        _normalize([counts[p] for p in pairs]),
    )


# ---------- Distances on the obstacle-free roadmap ----------


def roadmap_csr(roadmap):
    n = roadmap.graph.number_of_nodes()
    u, v, w = [], [], []
    for a, b, d in roadmap.graph.edges(data="weight"):
        u += [a, b]
        v += [b, a]
        w += [d, d]
    return csr_matrix((w, (u, v)), shape=(n, n))


class DistanceTable:
    """Dijkstra from every landmark on the obstacle-free roadmap (distances + predecessors)."""

    def __init__(self, landmark_nodes, dist, pred):
        self.landmark_nodes = np.asarray(landmark_nodes, dtype=int)
        self.dist = dist  # (L', N) for the unique landmark nodes
        self.pred = pred  # (L', N) predecessor node (-9999 = none)
        self.row = {int(n): i for i, n in enumerate(self.landmark_nodes)}

    @classmethod
    def compute(cls, roadmap, landmark_nodes):
        nodes = np.unique(np.asarray(landmark_nodes, dtype=int))
        dist, pred = dijkstra(roadmap_csr(roadmap), directed=False, indices=nodes, return_predecessors=True)
        return cls(nodes, dist, pred)

    @classmethod
    def load_or_compute(cls, roadmap, landmark_nodes, cache_dir=None):
        nodes = np.unique(np.asarray(landmark_nodes, dtype=int))
        if not cache_dir:
            return cls.compute(roadmap, nodes)
        lm_hash = hashlib.sha256(nodes.astype(np.int64).tobytes()).hexdigest()[:16]
        path = os.path.join(cache_dir, "dist_%s_%s.npz" % (roadmap.fingerprint[:16], lm_hash))
        if os.path.exists(path):
            try:
                z = np.load(path)
                if (int(z["version"]) == DIST_CACHE_VERSION and str(z["fingerprint"]) == roadmap.fingerprint
                        and np.array_equal(z["nodes"], nodes)):
                    return cls(nodes, z["dist"], z["pred"])
            except Exception as e:
                log.warning("distance table: could not read cache %s (%s); recomputing", path, e)
        table = cls.compute(roadmap, nodes)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npz")
            os.close(fd)
            np.savez_compressed(tmp, version=DIST_CACHE_VERSION, fingerprint=roadmap.fingerprint,
                                nodes=nodes, dist=table.dist, pred=table.pred)
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except OSError as e:
            log.warning("distance table: could not write cache %s (%s)", path, e)
        return table

    def path(self, s, t):
        """Free-graph shortest path s -> t as a node list (s must be a landmark node)."""
        pred = self.pred[self.row[s]]
        out = [t]
        while out[-1] != s:
            p = pred[out[-1]]
            if p < 0:
                return None
            out.append(int(p))
        return out[::-1]


# ---------- Importance ----------


class ImportanceEvaluator:
    """I_o of single obstacles on a fixed roadmap and trip set.

    Results are cached by (obstacle_id, blocked edge set, trip set): an obstacle is only
    re-evaluated if its own blocked edges change. Other obstacles never enter the computation.
    """

    def __init__(self, roadmap, trips, params=None, blocking_params=None, table=None, cache_dir=None):
        self.roadmap = roadmap
        self.trips = trips
        self.params = params or ImportanceParams()
        self.blocking_params = blocking_params or BlockingParams()
        self.table = table or DistanceTable.load_or_compute(roadmap, trips.landmark_nodes, cache_dir)
        self.trips_key = trips.key
        self.cache = {}  # (obstacle_id, frozenset(B), trips_key) -> ImportanceResult
        self.num_dijkstra_runs = 0  # blocked-graph Dijkstra runs (for tests / diagnostics)

        T = self.table
        self.s_nodes = trips.landmark_nodes[trips.sources]
        self.t_nodes = trips.landmark_nodes[trips.targets]
        self.s_rows = np.array([T.row[int(n)] for n in self.s_nodes], dtype=int)
        self.t_rows = np.array([T.row[int(n)] for n in self.t_nodes], dtype=int)
        self.d_free = T.dist[self.s_rows, self.t_nodes]
        reachable = np.isfinite(self.d_free)
        if not reachable.all():  # cannot happen on the largest component; be safe anyway
            log.warning("importance: %d trip(s) unreachable on the free roadmap; ignored", int((~reachable).sum()))
        self.valid = reachable

    # --- public API ---

    def blocked_edges_for(self, **footprint):
        cells = footprint_cells(self.roadmap, **footprint)
        return blocked_edges(self.roadmap, cells, self.blocking_params)

    def evaluate(self, obstacle_id, coarse_cells=None, fine_cells=None, polygon=None, square=None):
        """ImportanceResult for one obstacle footprint (see edge_blocking.footprint_cells)."""
        t0 = time.time()  # wall clock: compute time
        B = self.blocked_edges_for(coarse_cells=coarse_cells, fine_cells=fine_cells, polygon=polygon, square=square)
        return self.evaluate_blocked(obstacle_id, B, t0)

    def evaluate_square(self, obstacle_id, x, y, width):
        return self.evaluate(obstacle_id, square=(x, y, width))

    def forget(self, obstacle_id):
        for k in [k for k in self.cache if k[0] == obstacle_id]:
            del self.cache[k]

    def evaluate_blocked(self, obstacle_id, B, t0=None):
        t0 = time.time() if t0 is None else t0  # wall clock: compute time
        B = frozenset(B)
        n = len(self.trips)
        if not B:
            return ImportanceResult(0.0, 0.0, 0.0, 0.0, n, 0, B, time.time() - t0)  # wall clock: compute time
        key = (obstacle_id, B, self.trips_key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._compute(B, t0)
        self.cache[key] = result
        return result

    # --- internals ---

    def _affected(self, B):
        """Bool per trip: some blocked edge lies on a free shortest path s -> t."""
        D = self.table.dist
        tol = self.params.eps_path_rel * self.d_free + 1e-9
        hit = np.zeros(len(self.trips), dtype=bool)
        for u, v in B:
            w = self.roadmap.edge_weight(u, v)
            a = D[self.s_rows, u] + w + D[self.t_rows, v]
            b = D[self.s_rows, v] + w + D[self.t_rows, u]
            hit |= (np.abs(a - self.d_free) <= tol) | (np.abs(b - self.d_free) <= tol)
        return hit & self.valid

    def _blocked_lengths(self, weight, source, memo):
        if source not in memo:
            self.num_dijkstra_runs += 1
            memo[source] = nx.single_source_dijkstra_path_length(self.roadmap.graph, source, weight=weight)
        return memo[source]

    def _compute(self, B, t0):
        p = self.params
        w = self.trips.weights
        n = len(w)
        affected = np.nonzero(self._affected(B))[0]
        delta = np.zeros(n)
        delta_disc = np.zeros(n)
        if len(affected):
            # Edge filter instead of a graph copy: networkx skips edges whose weight is None.
            # (Same result as nx.restricted_view(G, [], B), ~4x faster than iterating a view.)
            hidden = set(B) | {(v, u) for u, v in B}

            def blocked_weight(u, v, d):
                return None if (u, v) in hidden else d["weight"]

            memo = {}
            for k in affected:  # memo groups trips by source: one Dijkstra per source
                s, t, d0 = int(self.s_nodes[k]), int(self.t_nodes[k]), self.d_free[k]
                db = self._blocked_lengths(blocked_weight, s, memo).get(t)
                delta[k] = p.d_max if db is None else min(max(db - d0, 0.0), p.d_max)
                delta_disc[k] = self._discovery_delta(blocked_weight, memo, B, s, t, d0, delta[k])
        tol = p.eps_path_rel * np.where(np.isfinite(self.d_free), self.d_free, 0.0) + 1e-9
        delta[delta <= tol] = 0.0
        delta_disc[delta_disc <= tol] = 0.0
        hit = delta > 0
        I_o = float(np.dot(w, delta))
        frac = float(w[hit].sum())
        return ImportanceResult(
            I_o=I_o,
            I_o_disc=float(np.dot(w, delta_disc)),
            frac_affected=frac,
            mean_detour_if_affected=I_o / frac if frac > 0 else 0.0,
            num_trips=n,
            num_affected=int(hit.sum()),
            blocked_edges=B,
            compute_time_s=time.time() - t0,  # wall clock: compute time
        )

    def _discovery_delta(self, weight, memo, B, s, t, d0, delta_k):
        """Extra distance when the obstacle is discovered at the first blocked edge of the path."""
        path = self.table.path(s, t)
        if path is None:
            return delta_k
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) in B or (b, a) in B:
                x = a  # endpoint of the first blocked edge nearer to s
                d_sx = self.table.dist[self.table.row[s], x]
                d_xt = self._blocked_lengths(weight, x, memo).get(t)
                if d_xt is None:
                    return self.params.d_max
                return min(max(d_sx + d_xt - d0, 0.0), self.params.d_max)
        return delta_k  # this shortest path avoids B (tie): no discovery detour


def _same_result(a, b):
    return (a.blocked_edges, a.I_o, a.I_o_disc, a.frac_affected) == (b.blocked_edges, b.I_o, b.I_o_disc, b.frac_affected)


class ObjectImportanceTracker:
    """Importance of the current ledger objects, each computed once.

    update() evaluates only objects that are new or whose footprint cells changed (the ledger
    refined their mean position or max width); an existing object whose footprint is unchanged
    is never touched, whatever else appears, moves or disappears. Belief plays no part.
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.entries = {}  # object_id -> (footprint key, ImportanceResult)
        self.objects = {}  # object_id -> (x, y, width), latest values (for drawing)

    def footprint_key(self, x, y, width):
        cells = footprint_cells(self.evaluator.roadmap, square=(x, y, width))
        return cells.tobytes()

    def update(self, objects):
        """objects: {object_id: (x, y, width)}. Returns the ids whose result changed or were removed."""
        changed = set()
        for oid in [oid for oid in self.entries if oid not in objects]:
            del self.entries[oid]
            self.evaluator.forget(oid)
            changed.add(oid)
        for oid, (x, y, width) in objects.items():
            key = self.footprint_key(x, y, width)
            old = self.entries.get(oid)
            if old is not None and old[0] == key:
                continue
            result = self.evaluator.evaluate_square(oid, x, y, width)
            self.entries[oid] = (key, result)
            if old is None or not _same_result(old[1], result):
                changed.add(oid)
        self.objects = {oid: tuple(v) for oid, v in objects.items()}
        return changed

    @property
    def results(self):
        return {oid: r for oid, (_k, r) in self.entries.items()}

    def get(self, object_id):
        entry = self.entries.get(object_id)
        return None if entry is None else entry[1]


def write_results_json(path, results, objects=None):
    """Write {object_id: result dict (+ x, y, width)} atomically."""
    out = {}
    for oid, r in results.items():
        d = r.to_dict()
        if objects and oid in objects:
            d["x"], d["y"], d["width"] = (float(v) for v in objects[oid])
        out[oid] = d
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
