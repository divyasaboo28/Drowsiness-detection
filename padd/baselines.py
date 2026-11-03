"""Baselines that PADD is compared against.

A  FixedThresholdSystem     - the classic tutorial rule, same numbers for everyone
B  PersonalThresholdSystem  - same rule, but with the user's calibrated eye scale
BS BlendshapeSystem         - MediaPipe's own eyeBlink / jawOpen scores, fixed cut-offs
"""

from __future__ import annotations

from . import config as C
from .features import FrameFeatures
from .profile import PersonalModel, UserProfile
from .system import ALARM, OK, DrowsinessSystem, Status


class _SustainedRule(DrowsinessSystem):
    """Alarm while `eye_closed` has been true for EYE_SECONDS, or for a few
    seconds after `mouth_open` has been true for YAWN_SECONDS."""

    eye_seconds = C.BASE_EYE_CLOSED_SECONDS

    def __init__(self):
        super().__init__()
        self.eye_since: float | None = None
        self.mouth_since: float | None = None
        self.yawn_alarm_until = -1.0

    def _measure(self, f: FrameFeatures) -> tuple[bool, bool, dict[str, float]]:
        raise NotImplementedError

    def _update(self, f: FrameFeatures | None, t: float) -> Status:
        if f is None:
            self.eye_since = self.mouth_since = None
            level = ALARM if t < self.yawn_alarm_until else OK
            return Status(level=level, score=0.0)

        eye_closed, mouth_open, info = self._measure(f)
        if eye_closed:
            self.eye_since = t if self.eye_since is None else self.eye_since
        else:
            self.eye_since = None
        if mouth_open:
            self.mouth_since = t if self.mouth_since is None else self.mouth_since
            if t - self.mouth_since >= C.BASE_YAWN_SECONDS:
                self.yawn_alarm_until = t + C.BASE_YAWN_ALARM_HOLD
        else:
            self.mouth_since = None

        closed_for = t - self.eye_since if self.eye_since is not None else 0.0
        reasons = []
        if closed_for >= self.eye_seconds:
            reasons.append(f"eyes closed {closed_for:.1f}s")
        if t < self.yawn_alarm_until:
            reasons.append("yawn")
        info["closed_for"] = closed_for
        return Status(
            level=ALARM if reasons else OK,
            score=min(1.0, closed_for / self.eye_seconds),
            reasons=reasons,
            info=info,
        )


class FixedThresholdSystem(_SustainedRule):
    name = f"Baseline (fixed EAR<{C.BASE_EAR_THRESHOLD})"
    short = "baseline_fixed"

    def _measure(self, f):
        return (
            f.ear < C.BASE_EAR_THRESHOLD,
            f.mar > C.BASE_MAR_THRESHOLD,
            {"ear": f.ear, "mar": f.mar, "ear_ref": C.BASE_EAR_THRESHOLD},
        )


class PersonalThresholdSystem(_SustainedRule):
    """Personal calibration only: no pose normalization, no fusion."""

    name = "Personal threshold only"
    short = "baseline_personal"

    def __init__(self, profile: UserProfile):
        super().__init__()
        self.profile = profile
        self.model = PersonalModel(profile, use_pose_norm=False)

    def _measure(self, f):
        openness = self.model.openness(f.ear, 0.0, 0.0)
        return (
            openness < C.OPEN_CLOSE_T,
            f.mar > self.profile.yawn_threshold,
            {"ear": f.ear, "mar": f.mar, "openness": openness},
        )


class BlendshapeSystem(_SustainedRule):
    name = "MediaPipe blendshapes (fixed)"
    short = "baseline_blendshape"

    def _measure(self, f):
        return (
            f.blink_bs > C.BS_BLINK_THRESHOLD,
            f.jaw_bs > C.BS_JAW_THRESHOLD,
            {"blink_bs": f.blink_bs, "jaw_bs": f.jaw_bs},
        )
