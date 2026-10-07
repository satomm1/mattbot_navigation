"""Unit tests for navigation_utils/object_layers.py: python3 -m pytest mattbot_navigation/test"""

import os
import sys

import numpy as np
import pytest

from navigation_utils.object_layers import BLOCKED, FREE, MIN_SIDE_M, footprint_cell_range, ledger_blockout_grid

RES = 0.05


def test_footprint_with_nonzero_origin():
    origin = (-99.6, -62.8)
    g = ledger_blockout_grid([(-90.0, -60.0, 0.5, 1.0)], 400, 200, RES, origin)
    rows, cols = np.nonzero(g == BLOCKED)
    xs = origin[0] + (cols + 0.5) * RES
    ys = origin[1] + (rows + 0.5) * RES
    assert xs.min() == pytest.approx(-90.25 + RES / 2, abs=RES) and xs.max() == pytest.approx(-89.75 - RES / 2, abs=RES)
    assert ys.min() == pytest.approx(-60.25 + RES / 2, abs=RES) and ys.max() == pytest.approx(-59.75 - RES / 2, abs=RES)
    assert set(np.unique(g)) == {BLOCKED, FREE}


def test_every_ledger_belief_is_blocked_removed_are_not():
    objs = [(1.0, 1.0, 0.4, 1.0), (2.0, 1.0, 0.4, 0.3), (3.0, 1.0, 0.4, 0.0), (4.0, 1.0, 0.4, -1.0)]
    g = ledger_blockout_grid(objs, 100, 40, RES, (0.0, 0.0))
    for x, _y, _w, b in objs:
        assert (g[20, int(x / RES)] == BLOCKED) == (b >= 0)


def test_minimum_footprint_and_off_map():
    g = ledger_blockout_grid([(1.0, 1.0, 0.0, 1.0)], 100, 100, RES, (0.0, 0.0))
    side = int(round(MIN_SIDE_M / RES))
    assert (g == BLOCKED).sum() >= side * side
    assert footprint_cell_range(50.0, 50.0, 0.5, RES, (0.0, 0.0), 100, 100) is None
    assert (ledger_blockout_grid([(50.0, 50.0, 0.5, 1.0)], 100, 100, RES, (0.0, 0.0)) == FREE).all()


def test_matches_belief_grid_footprint():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mattbot_dds", "src"))
    belief = pytest.importorskip("dds_utils.belief")

    class Obj:
        local_x, local_y, width = -3.21, 4.37, 0.6

    class State:
        obj, belief = Obj(), 0.4

    origin = (-10.0, -5.0)
    bg = belief.belief_grid([State()], 300, 300, RES, *origin)
    lg = ledger_blockout_grid([(Obj.local_x, Obj.local_y, Obj.width, 0.4)], 300, 300, RES, origin)
    assert np.array_equal(bg >= 0, lg == BLOCKED)
