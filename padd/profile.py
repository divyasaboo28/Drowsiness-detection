"""User profile (learned at calibration) and the runtime personal model.

The UserProfile stores all calibrated values as plain JSON — it's the
"fingerprint" of a user's eyes, blinks, mouth, and head orientation.
PersonalModel wraps a profile at runtime to convert raw measurements
into personalized ones (openness, relative pitch) and carries the
guarded online adaptation state, which is never saved to disk.

Author / Contributor: Praga
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as C

PROFILE_DIR = Path("profiles")


@dataclass
class UserProfile:
    name: str
    created: str
    ear_open: float  # median open-eye EAR looking straight ahead
    ear_open_mad: float
    ear_closed: float
    # EAR_open(dp, dy) = a + b*(dp/10) + c*(dp/10)^2 + d*(|dy|/10)
    pose_coef: list[float]
    pitch_sign: float  # multiplies raw pitch so that "head down" is positive
    pitch_neutral: float  # raw pitch of the user's normal posture
    yaw_neutral: float
    blink_dur_p95: float
    blink_dur_p99: float
    reopen_p95: float
    blink_rate: float  # blinks per minute at rest
    slow_blink_rate: float  # slow blinks per minute at rest (false-alarm floor)
    perclos_base: float
    mar_rest: float
    mar_talk_p95: float
    yawn_threshold: float
    warnings: list[str] = field(default_factory=list)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else profile_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "UserProfile":
        return cls(**json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"Profile '{self.name}'",
            f"  eye open EAR      {self.ear_open:.3f} (+/- {self.ear_open_mad:.3f})",
            f"  eye closed EAR    {self.ear_closed:.3f}",
            f"  pose model        {', '.join(f'{c:+.4f}' for c in self.pose_coef)}",
            f"  pitch sign        {self.pitch_sign:+.0f}",
            f"  blinks            {self.blink_rate:.1f}/min, p95 duration {self.blink_dur_p95:.2f}s,"
            f" p95 reopen {self.reopen_p95:.2f}s",
            f"  baseline PERCLOS  {self.perclos_base * 100:.1f}%",
            f"  mouth             rest {self.mar_rest:.2f}, talking p95 {self.mar_talk_p95:.2f},"
            f" yawn threshold {self.yawn_threshold:.2f}",
        ]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def profile_path(name: str) -> Path:
    return PROFILE_DIR / f"{name}.json"


def pose_design(dp: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Design matrix for the EAR-vs-pose model (shared by fitting and runtime)."""
    p = np.asarray(dp, dtype=float) / 10.0
    y = np.abs(np.asarray(dy, dtype=float)) / 10.0
    return np.column_stack([np.ones_like(p), p, p * p, y])


class PersonalModel:
    """Runtime wrapper that turns raw face measurements into personal coordinates.
    
    Key transformations:
    - rel_pitch/rel_yaw: subtract the user's neutral pose so 0 = facing camera
    - expected_open_ear: what the EAR SHOULD be at this head angle (pose model)
    - openness: how open the eye is relative to this person's range [0=closed, 1=open]
    - adapt: slow online drift correction for lighting/posture changes
    
    The pose model (a polynomial) was fitted during calibration by having
    the user look in different directions, learning how their EAR changes with
    head angle — this is what prevents false alarms when looking at a phone.
    """

    def __init__(self, profile: UserProfile, use_pose_norm: bool = True):
        self.p = profile
        self.use_pose_norm = use_pose_norm
        self.ear_offset = 0.0  # adaptation state
        self.pitch_offset = 0.0

    def rel_pitch(self, raw_pitch: float) -> float:
        """Convert raw pitch to person-relative degrees (positive = tilted downward)."""
        return self.p.pitch_sign * (raw_pitch - self.p.pitch_neutral) - self.pitch_offset

    def rel_yaw(self, raw_yaw: float) -> float:
        """Convert raw yaw to person-relative degrees (0 = centered)."""
        return raw_yaw - self.p.yaw_neutral

    def expected_open_ear(self, rel_pitch: float, rel_yaw: float) -> float:
        """Predict expected open-eye EAR at the current head orientation."""
        span = self.p.ear_open - self.p.ear_closed
        if self.use_pose_norm:
            dp = float(np.clip(rel_pitch, *C.POSE_PITCH_RANGE))
            x = pose_design(np.array([dp]), np.array([rel_yaw]))[0]
            ref = float(x @ np.asarray(self.p.pose_coef))
        else:
            ref = self.p.ear_open
        ref += self.ear_offset
        return float(np.clip(ref, self.p.ear_closed + 0.3 * span, self.p.ear_open + 0.5 * span))

    def openness(self, ear: float, rel_pitch: float, rel_yaw: float) -> float:
        """Scale EAR to [0, 1] using this person's closed and expected-open values.
        
        Values above 1.0 are possible (capped at 1.5) if eyes are wider than normal.
        """
        ref = self.expected_open_ear(rel_pitch, rel_yaw)
        return float(np.clip((ear - self.p.ear_closed) / (ref - self.p.ear_closed), 0.0, 1.5))

    def adapt(self, ear: float, raw_pitch: float, rel_yaw: float, dt: float) -> None:
        """Follow slow drift (lighting, posture). Call only while confidently alert."""
        span = self.p.ear_open - self.p.ear_closed
        rp = self.rel_pitch(raw_pitch)
        # Compute how far the current EAR is from what the model predicts.
        # Slowly shift the model toward this value (exponential moving average).
        residual = ear - self.expected_open_ear(rp, rel_yaw)
        a = 1.0 - math.exp(-dt / C.ADAPT_TAU_EAR)
        lim = C.ADAPT_MAX_EAR_FRAC * span
        self.ear_offset = float(np.clip(self.ear_offset + a * residual, -lim, lim))

        # Only adapt posture while the head is near level, so looking at a phone
        # never becomes the new "straight ahead".
        if abs(rp) < 8.0:
            a = 1.0 - math.exp(-dt / C.ADAPT_TAU_PITCH)
            self.pitch_offset = float(
                np.clip(self.pitch_offset + a * rp, -C.ADAPT_MAX_PITCH, C.ADAPT_MAX_PITCH)
            )
