#!/usr/bin/env python3
"""Offline detour decisions: for a path and a set of ledger objects, which detours are worth driving.

  rosrun mattbot_navigation plot_observation_detours.py \
      --map $(rospack find mattbot_mcl)/map_json/current_map.json \
      --objects objects.csv --start 10,18.9 --goal 80,18.9 --out detours.png \
      [--belief 0.0] [--beliefs box:0.3,cart:1.0] [--n-trips 5] [--margin 0] [--max-detour 15]

The path is the grid shortest path from --start to --goal on the roadmap C-space (or --path, a
CSV of x,y rows). Viewsheds use the same ray casting as the observation planner (other objects
block line of sight). I_o comes from the importance evaluator (uniform random landmarks), and
each object visible from no point of the path is scored V = q N p_gone I_o against
C = detour + v (turn + dwell), as in the observation planner. --objects: see
navigation_utils/map_io.load_objects.
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
from navigation_utils.obstacle_importance import ImportanceEvaluator, ImportanceParams, build_trip_set  # noqa: E402
from navigation_utils.observation_policy import (  # noqa: E402
    DetourValueParams,
    Candidate,
    ThresholdPolicy,
)
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
    ap.add_argument("--belief", type=float, default=0.0, help="belief of every object (0 = unknown, 1 = just seen)")
    ap.add_argument("--beliefs", default="", help="per-object beliefs, e.g. box:0.3,cart:1.0")
    ap.add_argument("--n-trips", type=float, default=5.0)
    ap.add_argument("--conclusive-prob", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--skip-start", type=float, default=1.0, help="no detour branching off this close to the start")
    ap.add_argument("--skip-goal", type=float, default=1.0, help="no detour branching off this close to the goal")
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
    views = {}
    for oid, (x, y, w) in objects.items():
        blockers, _sig = nearby_blockers(oid, (x, y), all_objects, args.r_max + 1.0)
        views[oid] = compute_viewshed(blocking, (x, y), origin, res, args.r_min, args.r_max, blockers=blockers)
    t1 = time.time()
    beliefs = {oid: args.belief for oid in objects}
    for item in filter(None, args.beliefs.split(",")):
        oid, b = item.split(":")
        beliefs[oid.strip()] = float(b)

    iparams = ImportanceParams()
    evaluator = ImportanceEvaluator(roadmap, build_trip_set(roadmap, iparams), iparams, cache_dir=args.cache_dir)
    importance = {oid: evaluator.evaluate_square(oid, x, y, w).I_o for oid, (x, y, w) in objects.items()}
    policy = ThresholdPolicy(
        cruise_speed=args.cruise_speed, dwell_s=args.dwell, importance_fn=lambda c: importance[c.object_id],
        importance_m_fn=lambda c: importance[c.object_id],
        detour_params=DetourValueParams(n_trips=args.n_trips, conclusive_prob=args.conclusive_prob,
                                        margin_m=args.margin, hard_cap_m=args.max_detour,
                                        skip_start_m=args.skip_start, skip_goal_m=args.skip_goal),
    )
    cands = [Candidate(oid, "object", x, y, beliefs[oid], w) for oid, (x, y, w) in objects.items()]
    t2 = time.time()
    dcands = policy.detour_candidates(path_xy, cands, views, now=0.0)
    targets = {c.object_id: (c.x, c.y, planner.viewpoint_nodes(views[c.object_id]), policy.max_detour_m(c))
               for c in dcands}
    targets = {oid: t for oid, t in targets.items() if t[3] > 0}
    detours = planner.compute(path_xy, targets, DetourParams(max_detour_m=args.max_detour, r_max=args.r_max))
    options = policy.opportunistic_options(path_xy, cands, views, now=0.0)
    chosen, evaluations = policy.detour_options(path_xy, cands, views, now=0.0, detours=detours)
    t3 = time.time()

    on_stop = {c.object_id for o in options for c in o.candidates}
    dc_ids = {c.object_id for c in dcands}
    ev_of = {}
    for ev in evaluations:
        for oid in ev.object_ids:
            if oid not in ev_of or ev.object_ids[0] == oid:
                ev_of[oid] = ev
    L = float(np.hypot(*np.diff(path_xy, axis=0).T).sum())
    print("path %.1f m, %d objects; viewsheds %.0f ms, detours %.1f ms; N=%g q=%g margin=%g m" % (
        L, len(objects), 1000 * (t1 - t0), 1000 * (t3 - t2), args.n_trips, args.conclusive_prob, args.margin))
    print("%-12s %5s %6s %6s %8s %7s %7s  %s" % ("object", "b", "I_o", "D*", "detour_m", "V_m", "C_m", "decision"))
    for oid in objects:
        c = next(u for u in cands if u.object_id == oid)
        ev, d_star = ev_of.get(oid), (policy.max_detour_m(c) if oid in dc_ids else None)
        if oid not in dc_ids:
            decision = "opportunistic stop" if oid in on_stop else "seen from path (opportunistic job)"
        elif d_star <= 0:
            decision = "never worth a detour (D* <= 0)"
        elif ev is None:
            decision = "no viewpoint within D*"
        else:
            decision = ("GO" if ev.chosen else "skip (%s)" % ev.blocked if ev.blocked else "skip") + (
                " (with %s)" % "+".join(o for o in ev.object_ids if o != oid) if len(ev.object_ids) > 1 else "")
        print("%-12s %5.2f %6.2f %6s %8s %7s %7s  %s" % (
            oid, beliefs[oid], importance[oid], "-" if d_star is None else "%.1f" % d_star,
            "-" if ev is None else "%.2f" % ev.detour_m, "-" if ev is None else "%.1f" % ev.value_m,
            "-" if ev is None else "%.1f" % ev.cost_m, decision))
    visible = [(oid, x, y) for oid, (x, y, _w) in objects.items() if oid not in dc_ids]
    options = options + chosen

    render_detour_png(args.out, roadmap, path_xy, detours, targets, options=options, occupancy=occupancy,
                      planner=planner, evaluations=evaluations, visible=visible, title="Observation detours: %s" % os.path.basename(args.map))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
