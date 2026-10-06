#!/usr/bin/env python3
"""Chooses where along a planned path to stop and check whether ledger objects are still there.

Inputs:   /map                  static map; line of sight is computed on it
          /object_beliefs       per-object belief (mattbot_dds object_belief_map.py)
          /observation/events   navigator observation progress (for the per-object cooldown)
Service:  /observation/select   (mattbot_dds/SelectObservations) path -> observation stops
Outputs:  /observation/viewsheds  OccupancyGrid: union of viewsheds of objects due a check (RViz)
          /observation/stops      MarkerArray: last selected stops and their targets (RViz)

Geometry is in navigation_utils/viewpoints.py, the check policy in observation_policy.py.
"""

import threading
import time

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
            turn_rate=float(rospy.get_param("~turn_rate", 1.0)),
            dwell_s=float(rospy.get_param("~dwell_s", 4.0)),
        )

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

    def beliefs_callback(self, msg):
        cands = [
            Candidate(o.object_id, o.class_name, o.local_x, o.local_y, o.belief, float(o.width))
            for o in msg.objects
        ]
        with self.lock:
            self.candidates = cands
            self.last_beliefs_time = time.time()

    def event_callback(self, msg):
        if msg.event == ObservationEvent.ENDED:
            with self.lock:
                self.policy.record_check(msg.object_id, msg.window_end or time.time())

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
            now = time.time()
            eligible = self.policy.eligible_candidates(self.candidates, now, exclude)
            self.ensure_viewsheds(eligible)
            views = {oid: vs for oid, (vs, _sig) in self.viewsheds.items()}
            options = self.policy.options(path_xy, self.candidates, views, now, exclude)

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
            eligible = self.policy.eligible_candidates(self.candidates, time.time())
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


if __name__ == "__main__":
    ObservationPlannerNode()
    rospy.spin()
