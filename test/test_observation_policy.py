"""Unit tests for navigation_utils/observation_policy.py: python3 -m pytest mattbot_navigation/test"""

import math

import numpy as np
import pytest

from navigation_utils.observation_policy import (
    OPPORTUNISTIC,
    REMAINING_PATH,
    Candidate,
    ThresholdPolicy,
    opportunistic_cost_s,
)
from navigation_utils.viewpoints import Stop, compute_viewshed

RES = 0.1


def cand(oid, belief, x=5.0, y=3.0):
    return Candidate(oid, "chair", x, y, belief)


def test_eligibility_threshold_exclude_cooldown():
    p = ThresholdPolicy(check_below_belief=0.5, cooldown_s=10.0)
    assert p.eligible(cand("a", 0.5), now=0.0)
    assert not p.eligible(cand("a", 0.6), now=0.0)
    assert not p.eligible(cand("a", 0.2), now=0.0, exclude={"a"})
    p.record_check("a", 100.0)
    assert not p.eligible(cand("a", 0.2), now=105.0)
    assert p.eligible(cand("a", 0.2), now=110.0)


def test_value_uses_belief():
    p = ThresholdPolicy()
    assert p.value(cand("a", 0.25)) == pytest.approx(0.75)


def test_value_uses_importance_fn():
    importance = {"a": 4.0, "b": 0.0}
    p = ThresholdPolicy(importance_fn=lambda c: importance[c.object_id])
    assert p.value(cand("a", 0.25)) == pytest.approx(3.0)
    assert p.value(cand("b", 0.25)) == 0.0
    # Importance does not change which objects are eligible
    assert p.eligible(cand("b", 0.25), now=0.0)


def test_cost_turn_out_dwell_and_back():
    # Path heading 0; one target straight left (+90 deg): turn 90 out, 90 back
    stop = Stop(0, 0.0, 0.0, 0.0, [("a", 0.0, 2.0)])
    assert opportunistic_cost_s(stop, turn_rate=1.0, dwell_s=3.0) == pytest.approx(math.pi + 3.0)
    # Two targets, right then left: 90 + 180 + 90 degrees of turning, two dwells
    stop = Stop(0, 0.0, 0.0, 0.0, [("b", 0.0, -2.0), ("a", 0.0, 2.0)])
    assert opportunistic_cost_s(stop, turn_rate=1.0, dwell_s=3.0) == pytest.approx(2 * math.pi + 6.0)


def setup_options(policy, candidates, now=0.0, exclude=()):
    path = np.column_stack([np.linspace(0, 10, 101), np.full(101, 1.0)])
    blocking = np.zeros((60, 100), dtype=bool)
    views = {c.object_id: compute_viewshed(blocking, (c.x, c.y), (0.0, 0.0), RES, 1.0, 3.5) for c in candidates}
    return policy.options(path, candidates, views, now, exclude)


def test_options_built_for_eligible_objects_only():
    cands = [cand("a", 0.2, x=2.0), cand("b", 0.9, x=8.0), cand("c", 0.4, x=8.0)]
    opts = setup_options(ThresholdPolicy(), cands, exclude={"a"})
    assert len(opts) == 1
    (opt,) = opts
    assert opt.kind == OPPORTUNISTIC and opt.resume == REMAINING_PATH
    assert [c.object_id for c in opt.candidates] == ["c"]
    assert opt.values == [pytest.approx(0.6)]
    assert opt.path_index == opt.stop.path_index and opt.cost_s > 0


def test_max_cost_drops_expensive_stops():
    cands = [cand("a", 0.2, x=2.0)]
    assert setup_options(ThresholdPolicy(max_cost_s=1.0), cands) == []
    assert len(setup_options(ThresholdPolicy(max_cost_s=0.0), cands)) == 1
