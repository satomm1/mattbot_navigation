#!/usr/bin/env python3
"""Chooses where along a planned path to stop and check whether ledger objects are still there.

Inputs:   /map                  static map; line of sight is computed on it
          /object_beliefs       per-object belief (mattbot_dds object_belief_map.py)
          /observation/events   navigator observation progress (for the per-object cooldown)
Service:  /observation/select   (mattbot_dds/SelectObservations) path -> observation stops
Outputs:  /observation/viewsheds  OccupancyGrid: union of viewsheds of objects due a check (RViz)
          /observation/stops      MarkerArray: last selected stops and their targets (RViz)

With ~importance_enabled, a background worker also builds (or loads from cache) the topological
roadmap of the static /map and computes each object's importance I_o once (expected extra travel
per trip if it blocks; navigation_utils/obstacle_importance.py). The policy value becomes
I_o * (1 - belief). Extra outputs:
          /observation/roadmap     MarkerArray: roadmap edges and nodes (latched)
          /observation/importance  MarkerArray: I_o per object and its blocked edges (latched)
          ~importance_png_path     PNG of the importance map (+ .json of the results)

With ~detour_enabled (implies importance), objects that no point of the path can see (visible ones
are left to opportunistic stops) may get one DETOUR: leave the path, look from the best viewpoint,
replan to the goal (navigation_utils/detour.py). It is returned only if its value of information
beats its cost, V - C > ~detour_margin_m (observation_policy.py), and the navigator drives it
(~observe_detour_enabled). Extra outputs:
          /observation/detours     MarkerArray: detour viewpoints, routes and extra distances
          ~detour_png_path         PNG of the last request's detours (+ .json)

Geometry is in navigation_utils/viewpoints.py, the check policy in observation_policy.py.
"""

import logging
import os
import threading
import time
import traceback

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from mattbot_dds.msg import ObjectBeliefArray, ObservationEvent, ObservationStop
from mattbot_dds.srv import SelectObservations, SelectObservationsResponse

from navigation_utils.observation_policy import Candidate, ThresholdPolicy
from navigation_utils.viewpoints import blocking_mask, compute_viewshed, nearby_blockers

RECOMPUTE_MOVE_M = 0.2  # recompute an object's viewshed if it moves further than this


class _RospyLogHandler(logging.Handler):
    """Forward library logging (navigation_utils.*) to rosout."""

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            rospy.logerr(msg)
        elif record.levelno >= logging.WARNING:
            rospy.logwarn(msg)
        else:
            rospy.loginfo(msg)


class ObservationPlannerNode:
    def __init__(self):
        rospy.init_node("observation_planner")
        self.r_min = float(rospy.get_param("~r_min", 1.0))
        self.r_max = float(rospy.get_param("~r_max", 3.5))
        # Other ledger objects block line of sight (a stop must not look through a known object)
        self.known_objects_block = bool(rospy.get_param("~known_objects_block", True))
        self.policy = ThresholdPolicy(
            check_below_belief=float(rospy.get_param("~check_below_belief", 0.5)),
            cooldown_s=float(rospy.get_param("~cooldown_s", 120.0)),
            max_cost_s=float(rospy.get_param("~max_cost_s", 0.0)),
            r_pref=float(rospy.get_param("~r_pref", 2.0)),
            skip_start_m=float(rospy.get_param("~skip_start_m", 0.5)),
            skip_goal_m=float(rospy.get_param("~skip_goal_m", 0.8)),
            merge_m=float(rospy.get_param("~merge_m", 0.5)),
            # Prefer stops from which the object is at most this far off the path heading (no overshoot)
            max_turn=np.radians(float(rospy.get_param("~max_turn_deg", 90.0))),
            turn_rate=float(rospy.get_param("~turn_rate", 1.0)),
            dwell_s=float(rospy.get_param("~dwell_s", 4.0)),
        )
        self.policy.cruise_speed = float(rospy.get_param("/cruising_velocity", 0.4))
        importance_enabled = bool(rospy.get_param("~importance_enabled", False))
        self.detour_enabled = bool(rospy.get_param("~detour_enabled", False))
        self.importance = None  # RoadmapWorker (roadmap of /map, importance, detour planner)
        if importance_enabled or self.detour_enabled:  # detours need I_o
            self.importance = RoadmapWorker(True, self.detour_enabled)
            self.policy.importance_fn = self.importance.importance_of
            self.policy.importance_m_fn = self.importance.importance_m_of
        self.detour_params = None
        self.viewpoints = {}  # object_id -> (Viewshed, DetourPlanner, viewpoint nodes)
        if self.detour_enabled:
            from navigation_utils.detour import DetourParams
            from navigation_utils.observation_policy import DetourValueParams

            max_detour_m = float(rospy.get_param("~max_detour_m", 15.0))
            self.detour_params = DetourParams(max_detour_m=max_detour_m, r_max=self.r_max)
            self.policy.detour_params = DetourValueParams(
                n_trips=float(rospy.get_param("~detour_n_trips", 5.0)),
                conclusive_prob=float(rospy.get_param("~detour_conclusive_prob", 1.0)),
                margin_m=float(rospy.get_param("~detour_margin_m", 0.0)),
                max_detours_per_path=int(rospy.get_param("~max_detours_per_path", 1)),
                hard_cap_m=max_detour_m,
                # No detour branching off the path this close to the robot (start) or the goal
                skip_start_m=float(rospy.get_param("~detour_skip_start_m", 1.0)),
                skip_goal_m=float(rospy.get_param("~detour_skip_goal_m", 1.0)),
            )
            self.detours_pub = rospy.Publisher("/observation/detours", MarkerArray, queue_size=1, latch=True)

        self.lock = threading.Lock()
        self.map_info = None
        self.map_header = None
        self.blocking = None
        self.viewsheds = {}  # object_id -> (Viewshed, blocker signature)
        self.candidates = []
        self.last_beliefs_time = None

        self.viewshed_pub = rospy.Publisher("/observation/viewsheds", OccupancyGrid, queue_size=1, latch=True)
        self.stops_pub = rospy.Publisher("/observation/stops", MarkerArray, queue_size=1, latch=True)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber("/object_beliefs", ObjectBeliefArray, self.beliefs_callback, queue_size=1)
        rospy.Subscriber("/observation/events", ObservationEvent, self.event_callback, queue_size=50)
        rospy.Service("/observation/select", SelectObservations, self.select_srv)
        rospy.Timer(rospy.Duration(2.0), self.viz_timer)
        rospy.Timer(rospy.Duration(30.0), self.health_timer)

    # ---------- Inputs ----------

    def map_callback(self, msg):
        blocking = blocking_mask(msg.data, msg.info.width, msg.info.height)
        with self.lock:
            self.map_info, self.map_header = msg.info, msg.header
            self.blocking = blocking
            self.viewsheds = {}  # map changed: line of sight may have changed
        rospy.loginfo("observation_planner: map %dx%d @ %.3f m", msg.info.width, msg.info.height, msg.info.resolution)
        if self.importance is not None:
            self.importance.set_map(msg)

    def beliefs_callback(self, msg):
        cands = [
            Candidate(o.object_id, o.class_name, o.local_x, o.local_y, o.belief, float(o.width))
            for o in msg.objects
        ]
        with self.lock:
            self.candidates = cands
            self.last_beliefs_time = rospy.get_time()
        if self.importance is not None:
            self.importance.set_objects({c.object_id: (c.x, c.y, c.width) for c in cands})

    def event_callback(self, msg):
        if msg.event == ObservationEvent.ENDED:
            with self.lock:
                self.policy.record_check(msg.object_id, msg.window_end or rospy.get_time())

    def health_timer(self, _event):
        if self.last_beliefs_time is None:
            rospy.logwarn_throttle(
                300, "observation_planner: no /object_beliefs yet (launch with ledger:=true object_belief:=true)"
            )

    # ---------- Viewsheds ----------

    def ensure_viewsheds(self, candidates):
        """Compute missing or stale viewsheds (call with self.lock held)."""
        info = self.map_info
        origin = (info.origin.position.x, info.origin.position.y)
        all_objects = [(o.object_id, o.x, o.y, o.width) for o in self.candidates]
        for c in candidates:
            blockers, signature = [], ()
            if self.known_objects_block:
                blockers, signature = nearby_blockers(c.object_id, (c.x, c.y), all_objects, self.r_max + 1.0)
            cached = self.viewsheds.get(c.object_id)
            if cached is not None:
                vs, sig = cached
                if sig == signature and np.hypot(vs.obj_xy[0] - c.x, vs.obj_xy[1] - c.y) <= RECOMPUTE_MOVE_M:
                    continue
            vs = compute_viewshed(
                self.blocking, (c.x, c.y), origin, info.resolution, self.r_min, self.r_max, blockers=blockers
            )
            self.viewsheds[c.object_id] = (vs, signature)
        # Forget objects that left the ledger
        live = {c.object_id for c in self.candidates}
        for oid in [oid for oid in self.viewsheds if oid not in live]:
            del self.viewsheds[oid]

    # ---------- Service ----------

    def select_srv(self, req):
        path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in req.path.poses], dtype=float)
        exclude = set(req.exclude_object_ids)
        with self.lock:
            if self.blocking is None or len(path_xy) < 2:
                return SelectObservationsResponse(stops=[])
            now = rospy.get_time()  # cooldowns use the navigator's window times (ROS time)
            eligible = self.policy.eligible_candidates(self.candidates, now, exclude)
            detour_eligible = []
            if self.detour_enabled:  # no belief threshold for detours: they need viewsheds too
                detour_eligible = [c for c in self.candidates if self.policy.detour_eligible(c, now, exclude)]
            self.ensure_viewsheds({c.object_id: c for c in eligible + detour_eligible}.values())
            views = {oid: vs for oid, (vs, _sig) in self.viewsheds.items()}
            options = self.policy.opportunistic_options(path_xy, self.candidates, views, now, exclude)
            evaluations, detours, targets = [], None, {}
            if self.detour_enabled:
                # Objects visible from any point of the path are left to opportunistic stops
                dcands = self.policy.detour_candidates(path_xy, self.candidates, views, now, exclude)
                detours, targets = self.compute_detours(path_xy, dcands, views)
                if detours is not None:
                    chosen, evaluations = self.policy.detour_options(
                        path_xy, self.candidates, views, now, exclude, detours
                    )
                    options += chosen
        if self.detour_enabled:
            self.log_detours(evaluations, targets)
            self.publish_detour_markers(evaluations, req.path.header.frame_id or "map")
            visible = [(c.object_id, c.x, c.y) for c in self.candidates
                       if c.object_id in views and c.object_id not in targets
                       and self.policy.checkable_from_path(path_xy, views[c.object_id])]
            self.importance.set_detours(path_xy, detours, targets, options, evaluations, visible)

        stops = []
        for opt in options:
            s = opt.stop
            stops.append(
                ObservationStop(
                    kind=opt.kind,
                    resume=opt.resume,
                    path_index=opt.path_index,
                    x=s.x,
                    y=s.y,
                    heading=s.heading,
                    object_ids=[c.object_id for c in opt.candidates],
                    targets=[Point(x=c.x, y=c.y) for c in opt.candidates],
                    beliefs=[c.belief for c in opt.candidates],
                    values=opt.values,
                    cost_s=opt.cost_s,
                )
            )
        if stops:
            rospy.loginfo(
                "observation_planner: %d stop(s) for %s",
                len(stops), ", ".join("+".join(s.object_ids) for s in stops),
            )
        self.publish_stop_markers(stops, req.path.header.frame_id or "map")
        return SelectObservationsResponse(stops=stops)

    # ---------- Detours ----------

    def compute_detours(self, path_xy, dcands, views):
        """Detours for detour candidates, each searched up to its own D* (call with self.lock
        held). Returns (detours, targets), detours None while the roadmap is not ready."""
        planner = self.importance.detour_planner
        if planner is None:
            return None, {}
        targets = {}
        for c in dcands:
            d_star = self.policy.max_detour_m(c)
            if d_star <= 0:
                continue  # not worth any detour (just seen, or blocks nothing): no search
            vs = views[c.object_id]
            cached = self.viewpoints.get(c.object_id)
            if cached is None or cached[0] is not vs or cached[1] is not planner:
                cached = (vs, planner, planner.viewpoint_nodes(vs))
                self.viewpoints[c.object_id] = cached
            targets[c.object_id] = (c.x, c.y, cached[2], d_star)
        for oid in [oid for oid in self.viewpoints if oid not in self.viewsheds]:
            del self.viewpoints[oid]
        t0 = time.time()  # wall clock: compute time
        detours = planner.compute(path_xy, targets, self.detour_params)
        rospy.logdebug("observation_planner: %d detour(s) for %d object(s) in %.0f ms",
                       len(detours), len(targets), 1000.0 * (time.time() - t0))  # wall clock: compute time
        return detours, targets

    def log_detours(self, evaluations, targets):
        for ev in sorted(evaluations, key=lambda e: -e.net_m):
            rospy.loginfo(
                "observation_planner: detour %s: +%.1f m, V=%.1f m, C=%.1f m -> %s",
                "+".join(ev.object_ids), ev.detour_m, ev.value_m, ev.cost_m,
                "GO" if ev.chosen else "skip (%s)" % ev.blocked if ev.blocked else "skip",
            )
        unreachable = sorted(set(targets) - {oid for ev in evaluations for oid in ev.object_ids})
        if unreachable:
            rospy.loginfo("observation_planner: no worthwhile viewpoint within D* for %s", ", ".join(unreachable))

    def publish_detour_markers(self, evaluations, frame_id):
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for k, ev in enumerate(evaluations):
            s = ev.option.stop
            rgb = (0.2, 0.5, 1.0) if ev.chosen else (0.6, 0.6, 0.6)
            m = Marker(ns="detour_viewpoints", id=k, type=Marker.SPHERE, action=Marker.ADD)
            m.header.frame_id = frame_id
            m.pose.position.x, m.pose.position.y = s.x, s.y
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            text = Marker(ns="detour_text", id=k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.header.frame_id = frame_id
            text.pose.position.x, text.pose.position.y, text.pose.position.z = s.x, s.y, 0.5
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = "%s +%.1f m V=%.1f C=%.1f %s" % (
                "+".join(ev.object_ids), ev.detour_m, ev.value_m, ev.cost_m,
                "GO" if ev.chosen else "skip (%s)" % ev.blocked if ev.blocked else "skip")
            rays = Marker(ns="detour_rays", id=k, type=Marker.LINE_LIST, action=Marker.ADD)
            rays.header.frame_id = frame_id
            rays.pose.orientation.w = 1.0
            rays.scale.x = 0.03
            rays.color.r, rays.color.g, rays.color.b = rgb
            rays.color.a = 0.8
            for c in ev.option.candidates:
                rays.points += [Point(x=s.x, y=s.y), Point(x=c.x, y=c.y)]
            markers.markers += [m, text, rays]
        self.detours_pub.publish(markers)

    # ---------- Visualization ----------

    def publish_stop_markers(self, stops, frame_id):
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for k, s in enumerate(stops):
            m = Marker(ns="observation_stops", id=2 * k, type=Marker.SPHERE, action=Marker.ADD)
            m.header.frame_id = frame_id
            m.pose.position.x, m.pose.position.y = s.x, s.y
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 1.0
            lines = Marker(ns="observation_rays", id=2 * k + 1, type=Marker.LINE_LIST, action=Marker.ADD)
            lines.header.frame_id = frame_id
            lines.pose.orientation.w = 1.0
            lines.scale.x = 0.03
            lines.color.r, lines.color.g, lines.color.b, lines.color.a = 1.0, 0.6, 0.0, 0.8
            for t in s.targets:
                lines.points += [Point(x=s.x, y=s.y), Point(x=t.x, y=t.y)]
            markers.markers += [m, lines]
        self.stops_pub.publish(markers)

    def viz_timer(self, _event):
        if self.viewshed_pub.get_num_connections() == 0:
            return
        with self.lock:
            if self.blocking is None:
                return
            eligible = self.policy.eligible_candidates(self.candidates, rospy.get_time())
            self.ensure_viewsheds(eligible)
            info, header = self.map_info, self.map_header
            grid = np.zeros((info.height, info.width), dtype=np.int8)
            for c in eligible:
                vs = self.viewsheds[c.object_id][0]
                rows, cols = vs.mask.shape
                grid[vs.j0:vs.j0 + rows, vs.i0:vs.i0 + cols][vs.mask] = 100
        msg = OccupancyGrid(info=info, data=grid.flatten().tolist())
        msg.header.frame_id = header.frame_id or "map"
        msg.header.stamp = rospy.Time.now()
        self.viewshed_pub.publish(msg)


class RoadmapWorker:
    """Roadmap of the static /map, per-object importance and the detour grid, on a background
    thread (importance is never computed on the service path).

    Compute-once rule: the roadmap and distance table depend only on the static /map (rebuilt
    only if its fingerprint changes); each object is evaluated alone against the obstacle-free
    roadmap, once, and again only if its footprint cells change (ObjectImportanceTracker).
    """

    def __init__(self, importance_enabled=True, detour_enabled=False):
        self.importance_enabled = importance_enabled
        self.detour_enabled = detour_enabled
        from navigation_utils.edge_blocking import BlockingParams
        from navigation_utils.obstacle_importance import ImportanceParams
        from navigation_utils.roadmap import DEFAULT_CACHE_DIR, RoadmapParams

        handler = _RospyLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        lib_log = logging.getLogger("navigation_utils")
        lib_log.addHandler(handler)
        lib_log.setLevel(logging.INFO)
        lib_log.propagate = False

        robot_radius = float(rospy.get_param("~robot_radius", 0.4))
        self.roadmap_params = RoadmapParams(
            roadmap_resolution=float(rospy.get_param("~roadmap_resolution", 0.2)),
            robot_radius=float(rospy.get_param("~roadmap_robot_radius", robot_radius)),
        )
        self.importance_params = ImportanceParams(
            d_max=float(rospy.get_param("~importance_d_max", 50.0)),
            trip_mode=str(rospy.get_param("~importance_trip_mode", "uniform_landmarks")),
            landmarks_file=os.path.expanduser(rospy.get_param("~landmarks_file", "")) or None,
            trip_counts_file=os.path.expanduser(rospy.get_param("~trip_counts_file", "")) or None,
            num_random_landmarks=int(rospy.get_param("~num_random_landmarks", 50)),
        )
        self.blocking_params = BlockingParams()
        self.cache_dir = os.path.expanduser(rospy.get_param("~roadmap_cache_dir", DEFAULT_CACHE_DIR))
        self.png_path = os.path.expanduser(
            rospy.get_param("~importance_png_path", os.path.join(self.cache_dir, "importance_latest.png"))
        )
        self.png_min_interval_s = float(rospy.get_param("~importance_png_min_interval_s", 10.0))
        self.detour_png_path = os.path.expanduser(
            rospy.get_param("~detour_png_path", os.path.join(self.cache_dir, "detour_latest.png"))
        )
        self.detour_png_min_interval_s = float(rospy.get_param("~detour_png_min_interval_s", 10.0))
        self.detour_planner = None
        self.detour_snapshot = None  # (path_xy, detours, targets, options) of the last request
        self.detour_png_dirty = False
        self.last_detour_png = 0.0

        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending_map = None
        self.objects = None  # latest {object_id: (x, y, width)} (local map frame)
        self.objects_dirty = False
        self.map_msg = None
        self.fingerprint = None
        self.roadmap = None
        self.trips = None
        self.tracker = None
        self.png_dirty = False
        self.last_png = 0.0

        self.importance_pub = None
        self.roadmap_pub = rospy.Publisher("/observation/roadmap", MarkerArray, queue_size=1, latch=True)
        if importance_enabled:
            self.importance_pub = rospy.Publisher("/observation/importance", MarkerArray, queue_size=1, latch=True)
        threading.Thread(target=self.run, name="importance_worker", daemon=True).start()

    # ---------- Inputs (any thread) ----------

    def set_map(self, msg):
        with self.lock:
            self.pending_map = msg
        self.wake.set()

    def set_detours(self, path_xy, detours, targets, options, evaluations=(), visible=()):
        """Last /observation/select result, for the detour PNG (drawn on the worker thread)."""
        if detours is None:
            return
        with self.lock:
            self.detour_snapshot = (path_xy, detours, targets, options, list(evaluations), list(visible))
            self.detour_png_dirty = True
        self.wake.set()

    def set_objects(self, objects):
        if not self.importance_enabled:
            return
        with self.lock:
            self.objects = objects
            self.objects_dirty = True
        self.wake.set()

    def importance_m_of(self, candidate):
        """I_o (m per trip), or None until the object has been evaluated (no detour before that)."""
        tracker = self.tracker
        result = tracker.get(candidate.object_id) if tracker is not None else None
        return None if result is None else result.I_o

    def importance_of(self, candidate):
        """I_o of an object (1.0 = neutral until it has been evaluated). A lookup, never a computation."""
        tracker = self.tracker
        result = tracker.get(candidate.object_id) if tracker is not None else None
        return 1.0 if result is None else result.I_o

    # ---------- Worker ----------

    def run(self):
        while not rospy.is_shutdown():
            self.wake.wait(timeout=1.0)
            self.wake.clear()
            with self.lock:
                map_msg, self.pending_map = self.pending_map, None
                objects = self.objects if (self.objects_dirty or map_msg is not None) else None
                self.objects_dirty = False
            try:
                if map_msg is not None:
                    self.load_roadmap(map_msg)
                if objects is not None and self.tracker is not None:
                    self.update_objects(objects)
                if self.png_dirty and time.time() - self.last_png >= self.png_min_interval_s:  # wall clock: CPU throttle
                    self.write_png()
                if self.detour_png_dirty and time.time() - self.last_detour_png >= self.detour_png_min_interval_s:  # wall clock: CPU throttle
                    self.write_detour_png()
            except Exception:  # keep the worker alive; the policy falls back to importance 1
                rospy.logerr("observation_planner: importance worker failed:\n%s", traceback.format_exc())

    def load_roadmap(self, msg):
        from navigation_utils.obstacle_importance import (
            ImportanceEvaluator, ObjectImportanceTracker, build_trip_set,
        )
        from navigation_utils.roadmap import load_or_build_roadmap, map_fingerprint

        data = np.asarray(msg.data, dtype=np.int8)
        info = msg.info
        origin = (info.origin.position.x, info.origin.position.y)
        fp = map_fingerprint(data, info.width, info.height, info.resolution, origin, self.roadmap_params)
        if fp == self.fingerprint:
            return  # same static map: keep the roadmap and every computed importance
        t0 = time.time()  # wall clock: compute time
        roadmap, cached = load_or_build_roadmap(
            data, info.width, info.height, info.resolution, origin, self.roadmap_params, self.cache_dir
        )
        self.map_msg, self.fingerprint, self.roadmap = msg, fp, roadmap
        if self.detour_enabled:
            from navigation_utils.detour import DetourPlanner

            self.detour_planner = DetourPlanner(roadmap)
        trips_info = ""
        if self.importance_enabled:
            trips = build_trip_set(roadmap, self.importance_params)
            evaluator = ImportanceEvaluator(
                roadmap, trips, self.importance_params, self.blocking_params, cache_dir=self.cache_dir
            )
            self.trips = trips
            self.tracker = ObjectImportanceTracker(evaluator)  # new map: all importances start over
            self.png_dirty = True
            trips_info = "; %d trips (%s)" % (len(trips), trips.mode)
        rospy.loginfo(
            "observation_planner: roadmap %d nodes, %d edges (%s, %.1f s)%s",
            roadmap.graph.number_of_nodes(), roadmap.graph.number_of_edges(),
            "cached" if cached else "built", time.time() - t0, trips_info,  # wall clock: compute time
        )
        self.publish_roadmap(msg.header.frame_id or "map")

    def update_objects(self, objects):
        changed = self.tracker.update(objects)
        if not changed:
            return
        for oid in sorted(changed):
            r = self.tracker.get(oid)
            if r is None:
                rospy.loginfo("observation_planner: importance: %s removed", oid)
                continue
            rospy.loginfo(
                "observation_planner: importance: %s I_o=%.2f m I_o_disc=%.2f m affected=%.0f%% "
                "(%d/%d trips) blocked_edges=%d (%.0f ms)",
                oid, r.I_o, r.I_o_disc, 100.0 * r.frac_affected, r.num_affected, r.num_trips,
                len(r.blocked_edges), 1000.0 * r.compute_time_s,
            )
        self.publish_importance(self.map_msg.header.frame_id or "map")
        self.png_dirty = True

    def write_png(self):
        from navigation_utils.importance_viz import render_importance_png
        from navigation_utils.obstacle_importance import write_results_json

        info = self.map_msg.info
        occupancy = (self.map_msg.data, info.width, info.height, info.resolution,
                     (info.origin.position.x, info.origin.position.y))
        results, objects = self.tracker.results, self.tracker.objects
        render_importance_png(self.png_path, self.roadmap, results, objects, trips=self.trips, occupancy=occupancy,
                              title="Obstacle importance (%s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
        write_results_json(os.path.splitext(self.png_path)[0] + ".json", results, objects)
        self.png_dirty = False
        self.last_png = time.time()  # wall clock: CPU throttle
        rospy.loginfo_throttle(60, "observation_planner: wrote %s" % self.png_path)

    def write_detour_png(self):
        from navigation_utils.importance_viz import render_detour_png

        with self.lock:
            snapshot, self.detour_png_dirty = self.detour_snapshot, False
        if snapshot is None or self.roadmap is None:
            return
        path_xy, detours, targets, options, evaluations, visible = snapshot
        info = self.map_msg.info
        occupancy = (self.map_msg.data, info.width, info.height, info.resolution,
                     (info.origin.position.x, info.origin.position.y))
        render_detour_png(self.detour_png_path, self.roadmap, path_xy, detours, targets, options=options,
                          occupancy=occupancy, planner=self.detour_planner, evaluations=evaluations,
                          visible=visible,
                          title="Observation detours (%s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
        write_detours_json(os.path.splitext(self.detour_png_path)[0] + ".json", detours, evaluations, visible)
        self.last_detour_png = time.time()  # wall clock: CPU throttle

    # ---------- RViz ----------

    def _edge_points(self, edges):
        rm = self.roadmap
        pts = []
        for e in edges:
            poly = rm.graph.edges[e]["polyline"]
            xs = rm.origin[0] + (poly[:, 1] + 0.5) * rm.res
            ys = rm.origin[1] + (poly[:, 0] + 0.5) * rm.res
            for k in range(len(poly) - 1):
                pts += [Point(x=xs[k], y=ys[k]), Point(x=xs[k + 1], y=ys[k + 1])]
        return pts

    def publish_roadmap(self, frame_id):
        rm = self.roadmap
        edges = Marker(ns="roadmap_edges", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        edges.header.frame_id = frame_id
        edges.pose.orientation.w = 1.0
        edges.scale.x = 0.03
        edges.color.r, edges.color.g, edges.color.b, edges.color.a = 0.55, 0.55, 0.55, 0.8
        edges.points = self._edge_points(rm.edge_list)
        nodes = Marker(ns="roadmap_nodes", id=1, type=Marker.SPHERE_LIST, action=Marker.ADD)
        nodes.header.frame_id = frame_id
        nodes.pose.orientation.w = 1.0
        nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.12
        nodes.color.r, nodes.color.g, nodes.color.b, nodes.color.a = 0.4, 0.4, 0.4, 0.9
        nodes.points = [Point(x=x, y=y) for x, y in rm.node_xy]
        self.roadmap_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL), edges, nodes]))

    def publish_importance(self, frame_id):
        markers = [Marker(action=Marker.DELETEALL)]
        for k, (oid, r) in enumerate(sorted(self.tracker.results.items())):
            x, y, _w = self.tracker.objects.get(oid, (0.0, 0.0, 0.0))
            text = Marker(ns="importance_text", id=k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.header.frame_id = frame_id
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, 0.6
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = "I_o %.1f / %.1f m (%.0f%%)" % (r.I_o, r.I_o_disc, 100.0 * r.frac_affected)
            markers.append(text)
            if r.blocked_edges:
                lines = Marker(ns="importance_blocked_edges", id=k, type=Marker.LINE_LIST, action=Marker.ADD)
                lines.header.frame_id = frame_id
                lines.pose.orientation.w = 1.0
                lines.scale.x = 0.08
                lines.color.r, lines.color.g, lines.color.b, lines.color.a = 0.9, 0.1, 0.1, 0.9
                lines.points = self._edge_points(sorted(r.blocked_edges))
                markers.append(lines)
        self.importance_pub.publish(MarkerArray(markers=markers))


def write_detours_json(path, detours, evaluations, visible=()):
    import json
    import tempfile

    out = {"detours": {oid: d.to_dict() for oid, d in detours.items()},
           "evaluations": [{"object_ids": ev.object_ids, "detour_m": ev.detour_m, "V_m": ev.value_m,
                            "C_m": ev.cost_m, "net_m": ev.net_m, "chosen": ev.chosen,
                            "viewpoint": [ev.option.stop.x, ev.option.stop.y], "values_m": ev.values_m}
                           for ev in evaluations],
           "opportunistic": [oid for oid, _x, _y in visible]}
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


if __name__ == "__main__":
    ObservationPlannerNode()
    rospy.spin()
