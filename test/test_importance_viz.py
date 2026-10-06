"""Unit tests for navigation_utils/importance_viz.py: python3 -m pytest mattbot_navigation/test"""

from PIL import Image

from navigation_utils.importance_viz import render_importance_png
from navigation_utils.obstacle_importance import ImportanceEvaluator, ImportanceParams, build_trip_set
from synthetic_maps import RES, ring, roadmap_of


def test_png_written(tmp_path):
    g = ring()
    rm = roadmap_of(g)
    params = ImportanceParams(num_random_landmarks=8)
    trips = build_trip_set(rm, params)
    ev = ImportanceEvaluator(rm, trips, params)
    objects = {"blocking": (6.0, 1.8, 0.8), "harmless": (6.0, 4.0, 0.5)}  # 2nd is inside the island
    results = {oid: ev.evaluate_square(oid, *xyw) for oid, xyw in objects.items()}
    assert results["blocking"].I_o > 0 and results["harmless"].I_o == 0

    out = tmp_path / "sub" / "importance.png"
    render_importance_png(str(out), rm, results, objects, trips=trips,
                          occupancy=(g.ravel(), g.shape[1], g.shape[0], RES, (0.0, 0.0)), dpi=50)
    with Image.open(out) as im:
        im.verify()
    with Image.open(out) as im:
        assert im.format == "PNG"
        assert im.size[0] == 600  # 12 in x 50 dpi
    assert not list(out.parent.glob("*.tmp*"))  # atomic write left nothing behind


def test_png_without_occupancy_or_results(tmp_path):
    rm = roadmap_of(ring())
    out = tmp_path / "empty.png"
    render_importance_png(str(out), rm, {}, {}, dpi=40)
    assert out.stat().st_size > 0
