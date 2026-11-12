"""Temporal event detectors: blinks, PERCLOS, yawns and head nods.

Each detector is a state machine that processes one frame at a time,
using timestamps in seconds so results don't depend on the camera's
frame rate. Passing None for a value means "no reliable measurement
this frame" and the detector simply waits.

The detectors convert continuous signals (eye openness, mouth ratio,
head pitch) into discrete events (BlinkEvent, YawnEvent, NodEvent)
that the fusion layer uses to build the drowsiness score.
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
    """Three-state blink detector: open -> closed -> reopening -> open.
    
    Classifies each eye closure as:
    - Normal blink: short duration, quick reopen
    - Slow blink: duration > personal p95 OR slow reopen (drowsiness indicator)
    - Long closure: eyes shut >= 1 second (strong drowsiness / microsleep)
    
    Uses hysteresis thresholds (OPEN_CLOSE_T / OPEN_REOPEN_T / OPEN_FULL_T)
    to avoid flickering between states due to noise.
    """

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
        """Returns elapsed seconds if currently in closed state, else 0.0."""
        return t - self.t_close if self.state == "closed" else 0.0

    def _finish(self, t: float) -> BlinkEvent:
        reopen = t - self.t_open
        long = self.duration >= self.long_closure
        slow = not long and (self.duration > self.slow_duration or reopen > self.slow_reopen)
        return BlinkEvent(t, self.duration, reopen, slow, long)

    def update(self, t: float, openness: float | None) -> list[BlinkEvent]:
        """Update detector with current eye openness [0..1] and timestamp t."""
        events: list[BlinkEvent] = []
        self.last_t = t
        if openness is None:
            return events
        if self.state == "open":
            # Transition OPEN -> CLOSED when eye openness drops below threshold
            if openness < C.OPEN_CLOSE_T:
                self.state, self.t_close = "closed", t
        elif self.state == "closed":
            # Transition CLOSED -> REOPENING when eye starts opening again
            if openness > C.OPEN_REOPEN_T:
                self.duration = t - self.t_close
                self.state, self.t_open = "reopening", t
        elif self.state == "reopening":
            # Either completes the blink (fully open / timeout) or re-closes
            if openness < C.OPEN_CLOSE_T:  # closed again before fully open
                events.append(self._finish(t))
                self.state, self.t_close = "closed", t
            elif openness >= C.OPEN_FULL_T or t - self.t_open >= C.REOPEN_TIMEOUT:
                events.append(self._finish(t))
                self.state = "open"
        return events


class PerclosTracker:
    """Tracks PERCLOS: percentage of time eyes are closed over a sliding window.
    
    PERCLOS is a well-established drowsiness metric from sleep research.
    A value above ~15% (baseline + margin) indicates the driver's eyes
    are spending too much time closed - a strong drowsiness signal.
    Uses a deque-based sliding window (default 60 seconds).
    """

    def __init__(self, window: float = C.PERCLOS_WINDOW):
        self.window = window
        self.samples: deque[tuple[float, bool]] = deque()
        self.n_closed = 0

    def update(self, t: float, openness: float | None) -> None:
        """Add a frame observation (openness < PERCLOS_T is counted as closed)."""
        if openness is not None:
            closed = openness < C.PERCLOS_T
            self.samples.append((t, closed))
            self.n_closed += closed
        # Evict samples outside the rolling time window
        while self.samples and self.samples[0][0] < t - self.window:
            _, c = self.samples.popleft()
            self.n_closed -= c

    def value(self) -> float:
        """Return the fraction of window frames where eyes were closed."""
        return self.n_closed / len(self.samples) if self.samples else 0.0


# --------------------------------------------------------------------------
# Yawns
# --------------------------------------------------------------------------
# Helper to distinguish yawns from speech: yawns are one smooth hump,
# while talking produces rapid zig-zag oscillations in mouth openness.
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
    """Detects yawns by finding sustained mouth-open events above a personal threshold.
    
    A valid yawn must:
    1. Exceed the calibrated MAR threshold (set safely above talking levels)
    2. Last at least YAWN_MIN_DURATION seconds (1.2s)
    3. NOT exhibit rapid oscillations (which indicate speech, not yawning)
    
    The oscillation filter counts direction changes in the MAR signal;
    speech produces >2 reversals/second, while yawns form a single smooth curve.
    """

    def __init__(self, threshold: float = C.DEFAULT_YAWN_THRESHOLD):
        self.threshold = threshold
        self.history: deque[tuple[float, float]] = deque()
        self.active = False
        self.t_start = 0.0
        self.peak = 0.0
        self.rejected_talking = 0

    def update(self, t: float, mar: float | None) -> YawnEvent | None:
        """Update with current mouth aspect ratio (MAR) and timestamp t."""
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

        # Yawn has ended: validate duration and oscillation profile
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
    """Detects drowsy head nods: slow forward droop followed by a fast recovery jerk.
    
    State machine: level -> droop -> recovering -> level
                   level -> lookdown -> level (deliberate, not counted)
    
    Key distinction from a normal look-down:
    - Drowsy nod: SLOW droop (< 20 deg/s) + FAST snap back (> 40 deg/s)
    - Voluntary look: FAST drop + steady hold -> ignored
    - Slump: SLOW droop that stays down -> slumped_for() grows (used by
      fusion.py for the head-slump ALARM)

    Uses exponentially smoothed pitch and velocity to filter sensor noise.
    """

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
        """Reset detector state to level."""
        self.state, self.p, self.vel, self.last_t, self.t_level = "level", None, 0.0, t, t

    def slumped_for(self, t: float) -> float:
        """Seconds the head has stayed down after a slow (drowsy) droop."""
        return t - self.t_droop if self.state == "droop" else 0.0

    def update(self, t: float, rel_pitch: float | None) -> NodEvent | None:
        """Process relative head pitch angle in degrees (positive = forward tilt)."""
        if rel_pitch is None:
            return None
        if self.p is None or self.last_t is None or t - self.last_t > 1.0:
            self.reset(t)
            self.p = rel_pitch
            return None
        dt = max(t - self.last_t, 1e-3)
        self.last_t = t
        prev = self.p
        # Exponential smoothing on head pitch position
        self.p += (1.0 - math.exp(-dt / C.NOD_SMOOTH_TAU)) * (rel_pitch - self.p)
        raw_vel = (self.p - prev) / dt
        # Exponential smoothing on angular velocity
        self.vel += (1.0 - math.exp(-dt / C.NOD_VEL_TAU)) * (raw_vel - self.vel)
        p = self.p

        if self.state == "level":
            if p < C.NOD_LEVEL:
                self.t_level = t
            elif p >= C.NOD_DROOP_MIN:
                elapsed = max(t - self.t_level, 1e-3)
                avg_vel = (C.NOD_DROOP_MIN - C.NOD_LEVEL) / elapsed
                # Slow droop suggests muscle tone loss from sleep onset
                if avg_vel <= C.NOD_SLOW_MAX_VEL:
                    self.state, self.t_droop, self.peak = "droop", t, p
                else:
                    self.state = "lookdown"
        elif self.state == "droop":
            self.peak = max(self.peak, p)
            # Rapid upward snap indicates sudden waking reflex
            if self.vel <= -C.NOD_RECOVERY_VEL:
                self.state, self.t_event = "recovering", t
                return NodEvent(t, self.peak)
            if p < C.NOD_LEVEL:
                self.state, self.t_level = "level", t
            # A slow droop that stays down is a slump, not a deliberate posture:
            # it stays in "droop" and slumped_for() keeps growing.
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
    """Tracks accumulated recent evidence for a specific drowsiness cue.
    
    Each detected event (e.g. slow blink) adds an exponentially decaying score:
    score = sum(exp(-(t - t_event) / tau)), normalized by the `full` count.
    Subtracts the user's natural baseline rate so active blinkers aren't penalized.
    """

    def __init__(self, tau: float, full: float, baseline_per_min: float = 0.0):
        self.tau = tau
        self.full = full
        self.baseline = baseline_per_min / 60.0 * tau  # steady-state value at rest
        self.events: deque[float] = deque()

    def add(self, t: float) -> None:
        """Register a new detected event at timestamp t."""
        self.events.append(t)

    def value(self, t: float) -> float:
        """Sum of exponential decay terms from past events within 5*tau."""
        while self.events and t - self.events[0] > 5.0 * self.tau:
            self.events.popleft()
        return sum(math.exp(-(t - te) / self.tau) for te in self.events)

    def score(self, t: float) -> float:
        """Normalized cue score in [0.0, 1.0] after baseline subtraction."""
        return float(np.clip((self.value(t) - self.baseline) / self.full, 0.0, 1.0))
