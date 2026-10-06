#!/usr/bin/env python3
"""Offline obstacle importance: build/load the roadmap, evaluate obstacles, print a table, write a PNG.

  rosrun mattbot_navigation plot_obstacle_importance.py \
      --map $(rospack find mattbot_mcl)/map_json/current_map.json \
      --objects objects.csv --out importance.png [--landmarks landmarks.yaml] \
      [--trip-mode weighted_landmarks --trip-counts counts.csv] [--robot-radius 0.4]

--map:      mattbot_mcl map JSON (current_map.json style) or a ROS map_server YAML (+ PGM).
--objects:  CSV rows x, y, width[, id] in the local map frame (header optional), or a JSON
            list of {object_id, x|local_x, y|local_y, width} (e.g. importance_latest.json).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from navigation_utils.edge_blocking import BlockingParams  # noqa: E402
from navigation_utils.importance_viz import render_importance_png  # noqa: E402
from navigation_utils.obstacle_importance import (  # noqa: E402
    TRIP_MODES,
    ImportanceEvaluator,
    ImportanceParams,
    build_trip_set,
    write_results_json,
)
from navigation_utils.roadmap import DEFAULT_CACHE_DIR, RoadmapParams, load_or_build_roadmap  # noqa: E402


def load_map(path):
    """-> (data, width, height, resolution, (origin_x, origin_y)) in ROS OccupancyGrid layout."""
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        d = d.get("data", {}).get("map", d)
        return d["occupancy"], d["width"], d["height"], d["resolution"], (d["origin_x"], d["origin_y"])
    import yaml
    from PIL import Image

    with open(path) as f:
        meta = yaml.safe_load(f)
    image = meta["image"]
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(path), image)
    pix = np.asarray(Image.open(image).convert("L"), dtype=float)
    p = pix / 255.0 if meta.get("negate", 0) else (255.0 - pix) / 255.0
    grid = np.full(pix.shape, -1, dtype=np.int8)
    grid[p > float(meta.get("occupied_thresh", 0.65))] = 100
    grid[p < float(meta.get("free_thresh", 0.196))] = 0
    grid = grid[::-1]  # image row 0 is the top of the map
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    return grid.ravel(), grid.shape[1], grid.shape[0], float(meta["resolution"]), (origin[0], origin[1])


def load_objects(path):
    """-> {object_id: (x, y, width)}"""
    objects = {}
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        items = data.items() if isinstance(data, dict) else enumerate(data)
        for key, o in items:
            oid = str(o.get("object_id", key))
            objects[oid] = (float(o.get("local_x", o.get("x"))), float(o.get("local_y", o.get("y"))),
                            float(o.get("width", 0.5)))
        return objects
    with open(path, newline="") as f:
        for k, rec in enumerate(csv.reader(f)):
            if not rec or rec[0].strip().startswith("#"):
                continue
            try:
                x, y, w = float(rec[0]), float(rec[1]), float(rec[2])
            except ValueError:
                continue  # header
            objects[rec[3].strip() if len(rec) > 3 and rec[3].strip() else "o%d" % k] = (x, y, w)
    return objects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", default="importance.png")
    ap.add_argument("--json-out", default=None, help="also write the results as JSON")
    ap.add_argument("--robot-radius", type=float, default=0.4)
    ap.add_argument("--roadmap-resolution", type=float, default=0.2)
    ap.add_argument("--trip-mode", choices=TRIP_MODES, default=TRIP_MODES[0])
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--trip-counts", default=None)
    ap.add_argument("--num-random-landmarks", type=int, default=50)
    ap.add_argument("--d-max", type=float, default=50.0)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    occupancy = load_map(args.map)
    t0 = time.time()
    params = RoadmapParams(roadmap_resolution=args.roadmap_resolution, robot_radius=args.robot_radius)
    roadmap, cached = load_or_build_roadmap(*occupancy, params=params, cache_dir=args.cache_dir)
    print("roadmap: %d nodes, %d edges, %s in %.2f s" % (
        roadmap.graph.number_of_nodes(), roadmap.graph.number_of_edges(),
        "loaded from cache" if cached else "built", time.time() - t0))

    iparams = ImportanceParams(d_max=args.d_max, trip_mode=args.trip_mode, landmarks_file=args.landmarks,
                               trip_counts_file=args.trip_counts, num_random_landmarks=args.num_random_landmarks)
    trips = build_trip_set(roadmap, iparams)
    evaluator = ImportanceEvaluator(roadmap, trips, iparams, BlockingParams(), cache_dir=args.cache_dir)

    objects = load_objects(args.objects)
    results = {oid: evaluator.evaluate_square(oid, *xyw) for oid, xyw in objects.items()}
    print("%-24s %8s %8s %7s %9s %8s %7s" % ("object", "I_o", "I_o_disc", "frac", "mean_det", "blocked", "ms"))
    for oid, r in sorted(results.items(), key=lambda kv: -kv[1].I_o):
        print("%-24s %8.2f %8.2f %6.1f%% %9.2f %8d %7.1f" % (
            oid, r.I_o, r.I_o_disc, 100 * r.frac_affected, r.mean_detour_if_affected, len(r.blocked_edges),
            1000 * r.compute_time_s))

    render_importance_png(args.out, roadmap, results, objects, trips=trips, occupancy=occupancy,
                          title="Obstacle importance: %s" % os.path.basename(args.map))
    print("wrote", args.out)
    if args.json_out:
        write_results_json(args.json_out, results, objects)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
