"""Unit tests for the scout-execution helpers in navigation_utils/observation_policy.py (localize_and_navigate
"Scouts", observation_planner ~scout_robot_heights): python3 -m pytest mattbot_navigation/test"""

from types import SimpleNamespace

import pytest

from navigation_utils.observation_policy import (
    AT_GOAL,
    DETOUR,
    SCOUT,
    parse_robot_heights,
    pick_scout,
    scout_deadline,
    scout_height_allowed,
    yields_to,
)


# ---------- Which robots scout (~scout_robot_heights vs ROBOT_HEIGHT) ----------

@pytest.mark.parametrize("text", ["", "any", "ALL", "*", " , "])
def test_any_height_allows_every_robot(text):
    heights = parse_robot_heights(text)
    assert heights is None
    for h in ("short", "tall", ""):
        assert scout_height_allowed(heights, h)


def test_listed_heights_only():
    short = parse_robot_heights("short")
    assert scout_height_allowed(short, "short")
    assert scout_height_allowed(short, " Short ")  # case and spaces do not matter
    assert not scout_height_allowed(short, "tall")
    assert not scout_height_allowed(short, "")  # ROBOT_HEIGHT unset: not allowed by a restricted list
    both = parse_robot_heights("short, tall")
    assert scout_height_allowed(both, "tall") and scout_height_allowed(both, "short")


# ---------- Deadline of a scout's way out ----------

def test_deadline_from_the_estimate_then_the_plan():
    est = scout_deadline(100.0, to_view_s=30.0, n_targets=2, dwell_s=4.0, factor=2.0)
    assert est == pytest.approx(100.0 + 2.0 * (30.0 + 8.0) + 20.0)
    # A longer planned drive extends it; a shorter one never shortens it
    longer = scout_deadline(101.0, 30.0, 2, 4.0, 2.0, plan_s=50.0, align_s=1.0, previous=est)
    assert longer == pytest.approx(101.0 + 2.0 * (50.0 + 1.0 + 8.0) + 20.0)
    shorter = scout_deadline(101.0, 30.0, 2, 4.0, 2.0, plan_s=5.0, align_s=1.0, previous=est)
    assert shorter >= est


# ---------- Which decision a navigator drives ----------

def decision(go=True, stamp=50.0, start=(1.0, 2.0), chosen=(False, True, True)):
    cands = [SimpleNamespace(object_ids=["o%d" % k], chosen=c) for k, c in enumerate(chosen)]
    return SimpleNamespace(go=go, header=SimpleNamespace(stamp=stamp), start=SimpleNamespace(x=start[0], y=start[1]),
                           candidates=cands)


def test_pick_scout_takes_the_first_chosen_candidate():
    cand, reason = pick_scout(decision(), now=55.0, pose_xy=(1.1, 2.0), last_stamp=0.0)
    assert reason == "" and cand.object_ids == ["o1"]  # o0 is not chosen; candidates come best net first


@pytest.mark.parametrize("kwargs,why", [
    (dict(d=None), "no decision"),
    (dict(d=decision(go=False)), "no go"),
    (dict(last_stamp=50.0), "already used"),  # a decision is driven once
    (dict(now=80.0), "too old"),  # a latched go from before a goal
    (dict(pose=(3.0, 2.0)), "another pose"),  # evaluated from somewhere else
    (dict(d=decision(chosen=(False, False))), "no chosen"),
])
def test_pick_scout_rejects(kwargs, why):
    d = kwargs.get("d", decision()) if "d" in kwargs else decision()
    cand, reason = pick_scout(d, now=kwargs.get("now", 55.0), pose_xy=kwargs.get("pose", (1.0, 2.0)),
                              last_stamp=kwargs.get("last_stamp", 0.0), max_age_s=12.0, max_offset_m=0.5)
    assert cand is None and why in reason


# ---------- En route: a scout gives way ----------

class I:  # an intent (ObservationIntent shape)
    def __init__(self, robot_id, eta, kind, committed=True):
        self.robot_id, self.eta, self.kind, self.committed = robot_id, eta, kind, committed


def test_scout_en_route_gives_way_to_goal_checks_and_detours_but_not_later_ones():
    assert yields_to(SCOUT, 100.0, 3, [I(7, 105.0, AT_GOAL)]).robot_id == 7  # someone looks anyway, soon enough
    assert yields_to(SCOUT, 100.0, 3, [I(7, 95.0, DETOUR)]).robot_id == 7
    assert yields_to(SCOUT, 100.0, 3, [I(7, 130.0, DETOUR)]) is None  # much later: keep scouting
    assert yields_to(SCOUT, 100.0, 9, [I(7, 104.0, SCOUT)]).robot_id == 7  # two scouts close in time: lower id
