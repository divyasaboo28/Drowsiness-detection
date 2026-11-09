"""PADD: personalized, pose-aware, multi-cue drowsiness detector.

Pipeline per frame:
  raw features -> personal normalization (profile + pose model)
  -> event detectors (blinks, PERCLOS, yawns, nods)
  -> decaying evidence per cue -> confidence-weighted fusion
  -> leaky integrator + hysteresis -> OK / CAUTION / ALARM
with a safety override for long eye closures and guarded online adaptation.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .detectors import (
    BlinkDetector,
    DecayingEvidence,
    NodDetector,
    PerclosTracker,
    YawnDetector,
)
from .features import FrameFeatures, size_confidence, yaw_confidence
from .profile import PersonalModel, UserProfile
from .system import ALARM, CAUTION, OK, DrowsinessSystem, Status


class PADDSystem(DrowsinessSystem):
    def __init__(
        self,
        profile: UserProfile,
        use_pose_norm: bool = True,
        use_nod: bool = True,
        use_adapt: bool = True,
        name: str = "PADD (personalized)",
        short: str = "padd",
    ):
        super().__init__()
        self.name, self.short = name, short
        self.profile = profile
        self.model = PersonalModel(profile, use_pose_norm=use_pose_norm)
        self.use_nod = use_nod
        self.use_adapt = use_adapt

        self.blinks = BlinkDetector(
            slow_duration=max(profile.blink_dur_p95, C.MIN_SLOW_BLINK_DURATION),
            slow_reopen=max(profile.reopen_p95, C.MIN_SLOW_REOPEN),
            long_closure=max(C.LONG_CLOSURE_MIN, profile.blink_dur_p99),
        )
        self.perclos = PerclosTracker()
        self.yawns = YawnDetector(profile.yawn_threshold)
        self.nods = NodDetector()

        self.ev_long = DecayingEvidence(C.TAU_LONG_CLOSURE, C.FULL_LONG_CLOSURE)
        self.ev_slow = DecayingEvidence(C.TAU_SLOW_BLINK, C.FULL_SLOW_BLINK, profile.slow_blink_rate)
        self.ev_yawn = DecayingEvidence(C.TAU_YAWN, C.FULL_YAWN)
        self.ev_nod = DecayingEvidence(C.TAU_NOD, C.FULL_NOD)

        self.rel = {"eye": 1.0, "blink": 1.0, "yawn": 1.0, "nod": 1.0}
        self.score = 0.0
        self.base_level = OK
        self.above_since: float | None = None
        self.last_t: float | None = None
        self.last_eye_t = -1e9
        self.last_event_t = -1e9  # time of the latest drowsy event of any kind
        self.last_rp: float | None = None  # last seen relative pitch
        self.head_down_since: float | None = None
        self.counts = {"slow_blinks": 0, "long_closures": 0, "yawns": 0, "nods": 0}

    # ------------------------------------------------------------------
    def _update(self, f: FrameFeatures | None, t: float) -> Status:
        dt = 0.0 if self.last_t is None else float(np.clip(t - self.last_t, 0.0, 0.5))
        self.last_t = t
        info: dict[str, float] = {}

        eye_conf = mouth_conf = 0.0
        openness = None
        if f is not None:
            rp = self.model.rel_pitch(f.pitch)
            ry = self.model.rel_yaw(f.yaw)
            eye_conf = yaw_confidence(ry) * size_confidence(f.eye_width_px)
            mouth_conf = yaw_confidence(ry)
            info.update(ear=f.ear, mar=f.mar, pitch=rp, yaw=ry,
                        ear_ref=self.model.expected_open_ear(rp, ry))

            if eye_conf >= C.MIN_CONFIDENCE:
                openness = self.model.openness(f.ear, rp, ry)
                info["openness"] = openness
            for ev in self.blinks.update(t, openness):
                if ev.long:
                    self.ev_long.add(t)
                    self.counts["long_closures"] += 1
                    self.last_event_t = t
                elif ev.slow:
                    self.ev_slow.add(t)
                    self.counts["slow_blinks"] += 1
                    self.last_event_t = t
            self.perclos.update(t, openness)

            if mouth_conf >= C.MIN_CONFIDENCE and self.yawns.update(t, f.mar):
                self.ev_yawn.add(t)
                self.counts["yawns"] += 1
                self.last_event_t = t
            if self.use_nod and self.nods.update(t, rp):
                self.ev_nod.add(t)
                self.counts["nods"] += 1
                self.last_event_t = t
            self.last_rp = rp
        else:
            self.perclos.update(t, None)

        # Head-down timer. If the face disappears while the head is down (it
        # often does when the chin drops to the chest), the head counts as
        # still down; if it disappears while level, the person just left.
        if self.last_rp is not None and self.last_rp >= C.NOD_DROOP_MIN:
            if self.head_down_since is None:
                self.head_down_since = t
        else:
            self.head_down_since = None
        head_down_for = 0.0 if self.head_down_since is None else t - self.head_down_since
        slumped_for = self.nods.slumped_for(t) if self.use_nod else 0.0

        # Cue reliability = smoothed landmark confidence
        a = 1.0 - math.exp(-dt / C.RELIABILITY_TAU) if dt > 0 else 0.0
        face = 1.0 if f is not None else 0.0
        for key, conf in (("eye", eye_conf), ("blink", eye_conf), ("yawn", mouth_conf), ("nod", face)):
            self.rel[key] += a * (conf - self.rel[key])

        # Cue scores (0..1)
        perclos = self.perclos.value()
        perclos_score = float(np.clip(
            (perclos - self.profile.perclos_base - C.PERCLOS_MARGIN) / C.PERCLOS_FULL_EXCESS, 0.0, 1.0))
        scores = {
            "eye": max(perclos_score, self.ev_long.score(t)),
            "blink": self.ev_slow.score(t),
            "yawn": self.ev_yawn.score(t),
            "nod": self.ev_nod.score(t) if self.use_nod else 0.0,
        }
        cues = {k: C.WEIGHTS[k] * self.rel[k] * s for k, s in scores.items()}
        fused = sum(cues.values())
        if dt > 0:
            self.score += (1.0 - math.exp(-dt / C.FUSION_TAU)) * (fused - self.score)

        # Hysteresis state machine. ALARM needs a high score AND fresh evidence:
        # the score alone remembers minutes of past events and would keep the
        # alarm on long after the person is alert again.
        D = self.score
        closed_for = self.blinks.closed_for(t)
        recent = t - self.last_event_t <= C.RECENT_WINDOW or closed_for >= 0.5
        info["recent"] = float(recent)
        if self.base_level == ALARM:
            if D < C.ALARM_OFF or not recent:
                self.base_level = CAUTION if D >= C.CAUTION_OFF else OK
                self.above_since = None
        else:
            if D > C.ALARM_ON and recent:
                self.above_since = t if self.above_since is None else self.above_since
                if t - self.above_since >= C.ALARM_HOLD:
                    self.base_level = ALARM
            else:
                self.above_since = None
            if self.base_level == CAUTION and D < C.CAUTION_OFF:
                self.base_level = OK
            elif self.base_level == OK and D > C.CAUTION_ON:
                self.base_level = CAUTION

        if openness is not None:
            self.last_eye_t = t
        # Override only while we can actually see the eyes (not after turning away)
        eyes_override = closed_for >= C.OVERRIDE_CLOSURE and t - self.last_eye_t < 0.5
        # Head overrides: a slow droop that stays down is the posture of
        # falling asleep; any head-down held very long means the eyes have
        # been off the road too long, whatever the reason.
        slump_override = slumped_for >= C.HEAD_SLUMP_TIME
        head_override = head_down_for >= C.HEAD_DOWN_ALARM
        override = eyes_override or slump_override or head_override
        level = ALARM if override else self.base_level

        # Guarded adaptation: only learn while clearly alert and eyes open
        if (self.use_adapt and f is not None and openness is not None and dt > 0
                and D < C.ADAPT_MAX_SCORE and openness > 0.6 and closed_for == 0.0):
            self.model.adapt(f.ear, f.pitch, self.model.rel_yaw(f.yaw), dt)

        reasons = []
        if eyes_override:
            reasons.append(f"eyes closed {closed_for:.1f}s")
        if slump_override:
            reasons.append(f"head slumped {slumped_for:.0f}s")
        elif head_override:
            reasons.append(f"head down {head_down_for:.0f}s")
        if scores["eye"] > 0.3:
            reasons.append(f"eyes: PERCLOS {perclos * 100:.0f}%" if perclos_score >= self.ev_long.score(t)
                           else "eyes: long closure")
        if scores["blink"] > 0.3:
            reasons.append("slow blinks")
        if scores["yawn"] > 0.3:
            reasons.append("yawning")
        if scores["nod"] > 0.3:
            reasons.append("head nodding")

        info.update(perclos=perclos, closed_for=closed_for, eye_conf=eye_conf,
                    head_down_for=head_down_for, slumped_for=slumped_for,
                    ear_offset=self.model.ear_offset, pitch_offset=self.model.pitch_offset)
        return Status(level=level, score=D, cues=cues, reasons=reasons, info=info)
