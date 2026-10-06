"""Load static maps and object lists for the offline tools (no ROS).

load_map:     mattbot_mcl map JSON (current_map.json style) or ROS map_server YAML (+ PGM/PNG)
              -> (data, width, height, resolution, (origin_x, origin_y)), ROS OccupancyGrid layout.
load_objects: CSV rows x, y, width[, id] (header optional) or a JSON list/dict of
              {object_id, x|local_x, y|local_y, width} -> {object_id: (x, y, width)}.
"""

import csv
import json
import os

import numpy as np


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
