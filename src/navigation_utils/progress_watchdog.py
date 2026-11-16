"""Progress watchdog for controllers that do not keep a trajectory's timetable.

The classical TrajectoryTracker is time-indexed, so the navigator detects a stuck robot by
missed waypoint deadlines (plan time > a waypoint's time). The learned local planner
(``local_planner:=qlp``) follows the path at its own pace and slows down for people, so
deadlines would trigger recovery for no reason. It is stuck only when its progress ALONG THE
PATH stops growing: less than ``min_progress_m`` gained over the last ``window_s`` seconds.
"""

from collections import deque


class ProgressWatchdog:
    """``update(t, progress_m)`` returns True once progress has stalled for a whole window."""

    def __init__(self, window_s=8.0, min_progress_m=0.2):
        if window_s <= 0.0 or min_progress_m <= 0.0:
            raise ValueError("window_s and min_progress_m must be positive")
        self.window_s = float(window_s)
        self.min_progress_m = float(min_progress_m)
        self._hist = deque()  # (t, progress_m), oldest first

    def reset(self):
        """A new plan (or a pause such as an observation stop) starts a fresh window."""
        self._hist.clear()

    def update(self, t, progress_m):
        t, p = float(t), float(progress_m)
        self._hist.append((t, p))
        # Keep exactly one sample at or before the window start, so the window is fully covered.
        while len(self._hist) >= 2 and self._hist[1][0] <= t - self.window_s:
            self._hist.popleft()
        t0, p0 = self._hist[0]
        if t - t0 < self.window_s:
            return False  # not watched for a whole window yet
        return p - p0 < self.min_progress_m
