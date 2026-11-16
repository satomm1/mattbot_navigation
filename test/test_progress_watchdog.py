"""Unit tests for navigation_utils/progress_watchdog.py: python3 -m pytest mattbot_navigation/test"""

import pytest

from navigation_utils.progress_watchdog import ProgressWatchdog


def _run(w, samples, dt=0.1):
    out = []
    for i, p in enumerate(samples):
        out.append(w.update(i * dt, p))
    return out


def test_steady_progress_is_never_stuck():
    w = ProgressWatchdog(window_s=2.0, min_progress_m=0.2)
    assert not any(_run(w, [0.03 * i for i in range(200)]))  # 0.3 m/s


def test_a_stall_is_reported_only_after_a_whole_window():
    w = ProgressWatchdog(window_s=2.0, min_progress_m=0.2)
    res = _run(w, [1.0] * 40)  # 4 s without moving
    assert not any(res[:20])  # less than a full window watched
    assert all(res[21:])


def test_slow_but_sufficient_progress_is_not_stuck():
    w = ProgressWatchdog(window_s=2.0, min_progress_m=0.2)
    assert not any(_run(w, [0.012 * i for i in range(100)]))  # 0.12 m/s: 0.24 m per window


def test_reset_starts_a_fresh_window():
    w = ProgressWatchdog(window_s=2.0, min_progress_m=0.2)
    assert _run(w, [0.0] * 30)[-1]
    w.reset()
    assert not w.update(10.0, 0.0)
    assert not w.update(11.0, 0.0)
    assert w.update(12.0, 0.0)


def test_parameters_must_be_positive():
    with pytest.raises(ValueError):
        ProgressWatchdog(window_s=0.0)
    with pytest.raises(ValueError):
        ProgressWatchdog(min_progress_m=-1.0)
