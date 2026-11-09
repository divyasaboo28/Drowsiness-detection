"""Common interface for every detector we compare (baselines and PADD)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import FrameFeatures

OK, CAUTION, ALARM = 0, 1, 2
LEVEL_NAMES = {OK: "OK", CAUTION: "CAUTION", ALARM: "ALARM"}


@dataclass
class Status:
    level: int = OK
    score: float = 0.0  # 0..1, what the system is "feeling"
    cues: dict[str, float] = field(default_factory=dict)  # per-cue contribution
    reasons: list[str] = field(default_factory=list)
    info: dict[str, float] = field(default_factory=dict)  # numbers for the overlay
    distracted: bool = False  # eyes off the road (separate from drowsiness level)


class DrowsinessSystem:
    name = "system"
    short = "sys"

    def __init__(self):
        self.alarm_count = 0
        self._prev_level = OK

    def update(self, f: FrameFeatures | None, t: float) -> Status:
        status = self._update(f, t)
        if status.level == ALARM and self._prev_level != ALARM:
            self.alarm_count += 1
        self._prev_level = status.level
        return status

    def _update(self, f: FrameFeatures | None, t: float) -> Status:
        raise NotImplementedError
