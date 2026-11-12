"""Common interface for every detector we compare (baselines and PADD).

This module defines the base class that all detection systems must inherit.
It provides a unified update() interface and automatic alarm counting,
so the evaluation framework can treat every system identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import FrameFeatures

# Drowsiness alert levels — used as integer constants throughout the system.
# OK=0 means the driver appears alert, CAUTION=1 is a warning state,
# and ALARM=2 triggers an audible/visual alert.
OK, CAUTION, ALARM = 0, 1, 2
LEVEL_NAMES = {OK: "OK", CAUTION: "CAUTION", ALARM: "ALARM"}


@dataclass
class Status:
    """Snapshot of a detection system's output for a single frame.
    
    Attributes:
        level:   Current alert level (OK / CAUTION / ALARM).
        score:   Fused drowsiness score in [0, 1] — higher means more drowsy.
        cues:    Per-cue weighted contributions (e.g. {"eye": 0.3, "yawn": 0.1}).
        reasons: Human-readable explanation strings shown in the UI overlay.
        info:    Raw numeric values for the on-screen debug display.
        distracted: True when the eyes are off the road (head down >= 3 s).
                 Separate from the drowsiness level: it never raises ALARM.
    """
    level: int = OK
    score: float = 0.0
    cues: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    info: dict[str, float] = field(default_factory=dict)
    distracted: bool = False


class DrowsinessSystem:
    """Abstract base class for all drowsiness detection systems.
    
    Subclasses implement _update() with their detection logic.
    The public update() wrapper automatically tracks alarm transitions
    and increments the alarm counter on each new ALARM onset.
    """
    name = "system"
    short = "sys"

    def __init__(self):
        self.alarm_count = 0       # Total number of distinct alarm episodes
        self._prev_level = OK      # Previous frame's level, for edge detection

    def update(self, f: FrameFeatures | None, t: float) -> Status:
        """Process one frame and return the current status.
        
        Counts a new alarm each time the level transitions INTO ALARM
        (rising edge only), so sustained alarms are counted once.
        """
        status = self._update(f, t)
        # Detect rising edge: level just became ALARM when it wasn't before
        if status.level == ALARM and self._prev_level != ALARM:
            self.alarm_count += 1
        self._prev_level = status.level
        return status

    def _update(self, f: FrameFeatures | None, t: float) -> Status:
        """Subclass hook: implement detection logic here."""
        raise NotImplementedError
