"""Baseline detection systems that PADD is compared against in evaluation.

Three baselines of increasing sophistication:
  A. FixedThresholdSystem:    Classic one-size-fits-all EAR < 0.22 rule
  B. PersonalThresholdSystem: Same rule but with calibrated eye scale
  BS. BlendshapeSystem:       Uses MediaPipe's own eyeBlink/jawOpen scores

All use the same simple logic: alarm if eyes closed for 1.6s or mouth
open for 1.0s. The comparison shows how much personalization + pose
normalization + multi-cue fusion improves over these simpler approaches.

Author / Contributor: Praga
"""

from __future__ import annotations

from . import config as C
from .features import FrameFeatures
from .profile import PersonalModel, UserProfile
from .system import ALARM, OK, DrowsinessSystem, Status


class _SustainedRule(DrowsinessSystem):
    """Base class for simple threshold-based detection.
    
    Logic: track how long the eyes have been closed and how long the
    mouth has been open. Trigger ALARM if either exceeds its time limit.
    After a yawn, the alarm stays active for YAWN_ALARM_HOLD seconds
    to avoid the alarm disappearing the instant the mouth closes.
    """

    eye_seconds = C.BASE_EYE_CLOSED_SECONDS

    def __init__(self):
        super().__init__()
        self.eye_since: float | None = None
        self.mouth_since: float | None = None
        self.yawn_alarm_until = -1.0

    def _measure(self, f: FrameFeatures) -> tuple[bool, bool, dict[str, float]]:
        """Return (eye_closed, mouth_open, debug_info_dict)."""
        raise NotImplementedError

    def _update(self, f: FrameFeatures | None, t: float) -> Status:
        if f is None:
            self.eye_since = self.mouth_since = None
            level = ALARM if t < self.yawn_alarm_until else OK
            return Status(level=level, score=0.0)

        eye_closed, mouth_open, info = self._measure(f)
        # Track continuous eye closure duration
        if eye_closed:
            self.eye_since = t if self.eye_since is None else self.eye_since
        else:
            self.eye_since = None
        # Track continuous mouth opening (yawn) duration
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
    """Baseline A: Fixed generic EAR threshold (0.22) and MAR threshold (0.5)."""
    name = f"Baseline (fixed EAR<{C.BASE_EAR_THRESHOLD})"
    short = "baseline_fixed"

    def _measure(self, f):
        return (
            f.ear < C.BASE_EAR_THRESHOLD,
            f.mar > C.BASE_MAR_THRESHOLD,
            {"ear": f.ear, "mar": f.mar, "ear_ref": C.BASE_EAR_THRESHOLD},
        )


class PersonalThresholdSystem(_SustainedRule):
    """Baseline B: Personal calibrated eye scale, but without head pose normalization."""

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
    """Baseline: Uses MediaPipe's built-in facial blendshapes (eyeBlink, jawOpen)."""
    name = "MediaPipe blendshapes (fixed)"
    short = "baseline_blendshape"

    def _measure(self, f):
        return (
            f.blink_bs > C.BS_BLINK_THRESHOLD,
            f.jaw_bs > C.BS_JAW_THRESHOLD,
            {"blink_bs": f.blink_bs, "jaw_bs": f.jaw_bs},
        )
