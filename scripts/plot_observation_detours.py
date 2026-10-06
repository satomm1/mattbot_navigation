#!/usr/bin/env python3
"""Offline observation detours: for a path and a set of objects, the extra distance to see each one.

  rosrun mattbot_navigation plot_observation_detours.py \
      --map $(rospack find mattbot_mcl)/map_json/current_map.json \
      --objects objects.csv --start 10,18.9 --goal 80,18.9 --out detours.png [--max-detour 15]

The path is the grid shortest path from --start to --goal on the roadmap C-space (or --path, a
CSV of x,y rows). Viewsheds use the same ray casting as the observation planner (other objects
block line of sight). --objects: see navigation_utils/map_io.load_objects.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from navigation_utils.detour import DetourParams, DetourPlanner  # noqa: E402
from navigation_utils.importance_viz import render_detour_png  # noqa: E402
from navigation_utils.map_io import load_map, load_objects  # noqa: E402
from navigation_utils.observation_policy import DETOUR, Candidate, ThresholdPolicy  # noqa: E402
from navigation_utils.roadmap import DEFAULT_CACHE_DIR, RoadmapParams, load_or_build_roadmap  # noqa: E402
from navigation_utils.viewpoints import blocking_mask, compute_viewshed, nearby_blockers  # noqa: E402


def xy_arg(text):
    x, y = text.split(",")
    return float(x), float(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--start", type=xy_arg)
    ap.add_argument("--goal", type=xy_arg)
    ap.add_argument("--path", help="CSV of x,y path points (instead of --start/--goal)")
    ap.add_argument("--out", default="detours.png")
    ap.add_argument("--max-detour", type=float, default=15.0)
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=3.5)
    ap.add_argument("--robot-radius", type=float, default=0.4)
    ap.add_argument("--cruise-speed", type=float, default=0.4)
    ap.add_argument("--dwell", type=float, default=4.0)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    occupancy = load_map(args.map)
    data, width, height, res, origin = occupancy
    roadmap, _cached = load_or_build_roadmap(*occupancy, params=RoadmapParams(robot_radius=args.robot_radius),
                                             cache_dir=args.cache_dir)
    planner = DetourPlanner(roadmap)
    if args.path:
        path_xy = np.loadtxt(args.path, delimiter=",", ndmin=2)[:, :2]
    elif args.start and args.goal:
        path_xy = planner.shortest_path_xy(args.start, args.goal)
        if path_xy is None:
            sys.exit("no path from %s to %s" % (args.start, args.goal))
    else:
        sys.exit("give --path or --start and --goal")

    blocking = blocking_mask(data, width, height)
    objects = load_objects(args.objects)
    all_objects = [(oid, x, y, w) for oid, (x, y, w) in objects.items()]
    t0 = time.time()
    views, targets = {}, {}
    for oid, (x, y, w) in objects.items():
        blockers, _sig = nearby_blockers(oid, (x, y), all_objects, args.r_max + 1.0)
        views[oid] = compute_viewshed(blocking, (x, y), origin, res, args.r_min, args.r_max, blockers=blockers)
        targets[oid] = (x, y, planner.viewpoint_nodes(views[oid]))
    t1 = time.time()
    detours = planner.compute(path_xy, targets, DetourParams(max_detour_m=args.max_detour, r_max=args.r_max))
    t2 = time.time()

    policy = ThresholdPolicy(check_below_belief=1.0, cruise_speed=args.cruise_speed, dwell_s=args.dwell)
    cands = [Candidate(oid, "object", x, y, 0.0, w) for oid, (x, y, w) in objects.items()]
    options = policy.options(path_xy, cands, views, now=0.0, detours=detours)
    on_stop = {c.object_id for o in options if o.kind != DETOUR for c in o.candidates}
    det_cost = {o.candidates[0].object_id: o.cost_s for o in options if o.kind == DETOUR}

    L = float(np.hypot(*np.diff(path_xy, axis=0).T).sum())
    print("path %.1f m, %d objects; viewsheds %.0f ms, detours %.1f ms" % (
        L, len(objects), 1000 * (t1 - t0), 1000 * (t2 - t1)))
    print("%-12s %10s %10s %8s  %s" % ("object", "detour_m", "to_view_m", "cost_s", "status"))
    for oid in objects:
        d = detours.get(oid)
        if oid in on_stop:
            status = "opportunistic stop on the path"
        elif d is None:
            status = "skipped (no viewpoint within %.0f m detour)" % args.max_detour
        elif d.on_path:
            status = "visible from the path (no usable stop there)"
        else:
            status = "DETOUR via (%.1f, %.1f), leave at path index %d" % (*d.viewpoint_xy, d.leave_index)
        print("%-12s %10s %10s %8s  %s" % (
            oid, "-" if d is None else "%.2f" % d.detour_m, "-" if d is None else "%.2f" % d.to_view_m,
            "%.1f" % det_cost[oid] if oid in det_cost else "-", status))

    render_detour_png(args.out, roadmap, path_xy, detours, targets, options=options, occupancy=occupancy,
                      planner=planner, title="Observation detours: %s" % os.path.basename(args.map))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
