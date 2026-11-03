"""Temporal event detectors: blinks, PERCLOS, yawns and head nods.

Every detector takes timestamps in seconds, so results do not depend on the
camera's frame rate. Passing None for a value means "no reliable measurement
this frame" and the detector simply waits.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Blinks and eye closures
# --------------------------------------------------------------------------
@dataclass
class BlinkEvent:
    t: float  # time the event was completed
    duration: float  # closed time (close -> reopen crossing)
    reopen: float  # time to go from half-open to fully open
    slow: bool
    long: bool  # long closure (>= LONG_CLOSURE_MIN), not counted as a blink


class BlinkDetector:
    def __init__(
        self,
        slow_duration: float = C.DEFAULT_BLINK_DUR_P95,
        slow_reopen: float = C.DEFAULT_REOPEN_P95,
        long_closure: float = C.LONG_CLOSURE_MIN,
    ):
        self.slow_duration = slow_duration
        self.slow_reopen = slow_reopen
        self.long_closure = long_closure
        self.state = "open"
        self.t_close = 0.0
        self.t_open = 0.0
        self.duration = 0.0
        self.last_t = 0.0

    def closed_for(self, t: float) -> float:
        return t - self.t_close if self.state == "closed" else 0.0

    def _finish(self, t: float) -> BlinkEvent:
        reopen = t - self.t_open
        long = self.duration >= self.long_closure
        slow = not long and (self.duration > self.slow_duration or reopen > self.slow_reopen)
        return BlinkEvent(t, self.duration, reopen, slow, long)

    def update(self, t: float, openness: float | None) -> list[BlinkEvent]:
        events: list[BlinkEvent] = []
        self.last_t = t
        if openness is None:
            return events
        if self.state == "open":
            if openness < C.OPEN_CLOSE_T:
                self.state, self.t_close = "closed", t
        elif self.state == "closed":
            if openness > C.OPEN_REOPEN_T:
                self.duration = t - self.t_close
                self.state, self.t_open = "reopening", t
        elif self.state == "reopening":
            if openness < C.OPEN_CLOSE_T:  # closed again before fully open
                events.append(self._finish(t))
                self.state, self.t_close = "closed", t
            elif openness >= C.OPEN_FULL_T or t - self.t_open >= C.REOPEN_TIMEOUT:
                events.append(self._finish(t))
                self.state = "open"
        return events


class PerclosTracker:
    """Fraction of recent time with eyes at least 80% closed."""

    def __init__(self, window: float = C.PERCLOS_WINDOW):
        self.window = window
        self.samples: deque[tuple[float, bool]] = deque()
        self.n_closed = 0

    def update(self, t: float, openness: float | None) -> None:
        if openness is not None:
            closed = openness < C.PERCLOS_T
            self.samples.append((t, closed))
            self.n_closed += closed
        while self.samples and self.samples[0][0] < t - self.window:
            _, c = self.samples.popleft()
            self.n_closed -= c

    def value(self) -> float:
        return self.n_closed / len(self.samples) if self.samples else 0.0


# --------------------------------------------------------------------------
# Yawns
# --------------------------------------------------------------------------
def direction_changes(values: np.ndarray, amplitude: float) -> int:
    """Count reversals bigger than `amplitude` (zig-zags of a talking mouth)."""
    if len(values) < 2:
        return 0
    changes = 0
    direction = 0  # +1 rising, -1 falling, 0 not yet known
    lo = hi = extreme = values[0]
    for v in values[1:]:
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if v - lo > amplitude:
                direction, extreme = 1, v
            elif hi - v > amplitude:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v > amplitude:
                changes += 1
                direction, extreme = -1, v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme > amplitude:
                changes += 1
                direction, extreme = 1, v
    return changes


@dataclass
class YawnEvent:
    t: float
    duration: float
    peak: float


class YawnDetector:
    def __init__(self, threshold: float = C.DEFAULT_YAWN_THRESHOLD):
        self.threshold = threshold
        self.history: deque[tuple[float, float]] = deque()
        self.active = False
        self.t_start = 0.0
        self.peak = 0.0
        self.rejected_talking = 0

    def update(self, t: float, mar: float | None) -> YawnEvent | None:
        if mar is None:
            return None
        self.history.append((t, mar))
        while self.history and self.history[0][0] < t - 15.0:
            self.history.popleft()

        if not self.active:
            if mar > self.threshold:
                self.active, self.t_start, self.peak = True, t, mar
            return None

        self.peak = max(self.peak, mar)
        if mar >= self.threshold * C.YAWN_END_RATIO:
            return None

        self.active = False
        duration = t - self.t_start
        if duration < C.YAWN_MIN_DURATION:
            return None
        # A yawn is one slow hump; speech zig-zags. Look at the event plus 1 s before.
        win = np.array([m for (ts, m) in self.history if ts >= self.t_start - 1.0])
        span = duration + 1.0
        rate = direction_changes(win, C.YAWN_OSC_AMPLITUDE) / span
        if rate > C.YAWN_MAX_OSC_RATE:
            self.rejected_talking += 1
            return None
        return YawnEvent(t, duration, self.peak)


# --------------------------------------------------------------------------
# Head nods
# --------------------------------------------------------------------------
@dataclass
class NodEvent:
    t: float
    depth: float  # degrees the head had dropped


class NodDetector:
    """Separates a drowsy nod (slow droop, then a fast jerk back up) from a
    deliberate look-down (fast drop, then holding still)."""

    def __init__(self):
        self.state = "level"
        self.p = None  # smoothed relative pitch (positive = down)
        self.vel = 0.0
        self.last_t = None
        self.t_level = 0.0
        self.t_droop = 0.0
        self.peak = 0.0
        self.t_event = 0.0

    def reset(self, t: float) -> None:
        self.state, self.p, self.vel, self.last_t, self.t_level = "level", None, 0.0, t, t

    def update(self, t: float, rel_pitch: float | None) -> NodEvent | None:
        if rel_pitch is None:
            return None
        if self.p is None or self.last_t is None or t - self.last_t > 1.0:
            self.reset(t)
            self.p = rel_pitch
            return None
        dt = max(t - self.last_t, 1e-3)
        self.last_t = t
        prev = self.p
        self.p += (1.0 - math.exp(-dt / C.NOD_SMOOTH_TAU)) * (rel_pitch - self.p)
        raw_vel = (self.p - prev) / dt
        self.vel += (1.0 - math.exp(-dt / C.NOD_VEL_TAU)) * (raw_vel - self.vel)
        p = self.p

        if self.state == "level":
            if p < C.NOD_LEVEL:
                self.t_level = t
            elif p >= C.NOD_DROOP_MIN:
                elapsed = max(t - self.t_level, 1e-3)
                avg_vel = (C.NOD_DROOP_MIN - C.NOD_LEVEL) / elapsed
                if avg_vel <= C.NOD_SLOW_MAX_VEL:
                    self.state, self.t_droop, self.peak = "droop", t, p
                else:
                    self.state = "lookdown"
        elif self.state == "droop":
            self.peak = max(self.peak, p)
            if self.vel <= -C.NOD_RECOVERY_VEL:
                self.state, self.t_event = "recovering", t
                return NodEvent(t, self.peak)
            if p < C.NOD_LEVEL:
                self.state, self.t_level = "level", t
            elif t - self.t_droop > C.NOD_MAX_DROOP_TIME:
                self.state = "lookdown"
        elif self.state == "lookdown":
            if p < C.NOD_LEVEL:
                self.state, self.t_level = "level", t
        elif self.state == "recovering":
            if p < C.NOD_DROOP_MIN * 0.5 or t - self.t_event > 2.0:
                self.state, self.t_level = "level", t
        return None


# --------------------------------------------------------------------------
# Decaying evidence
# --------------------------------------------------------------------------
class DecayingEvidence:
    """Each event contributes exp(-age / tau). Score saturates at `full`
    events, after subtracting what the user produces at rest."""

    def __init__(self, tau: float, full: float, baseline_per_min: float = 0.0):
        self.tau = tau
        self.full = full
        self.baseline = baseline_per_min / 60.0 * tau  # steady-state value at rest
        self.events: deque[float] = deque()

    def add(self, t: float) -> None:
        self.events.append(t)

    def value(self, t: float) -> float:
        while self.events and t - self.events[0] > 5.0 * self.tau:
            self.events.popleft()
        return sum(math.exp(-(t - te) / self.tau) for te in self.events)

    def score(self, t: float) -> float:
        return float(np.clip((self.value(t) - self.baseline) / self.full, 0.0, 1.0))
