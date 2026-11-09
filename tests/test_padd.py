"""Tests on a synthetic person, so the logic can be checked without a webcam.

The synthetic person has narrow eyes (open EAR 0.24) whose EAR drops to 0.14
when looking down at a phone - exactly the case where the fixed 0.22 rule
raises false alarms. Their raw pitch also runs the "wrong" way (sign -1),
which calibration has to discover.
"""

import math

import numpy as np
import pytest

from padd import config as C
from padd.baselines import FixedThresholdSystem
from padd.calibration import PHASES, fit_profile
from padd.detectors import NodDetector, YawnDetector
from padd.evaluate import SegmentRow, score_run
from padd.features import FrameFeatures
from padd.fusion import PADDSystem
from padd.profile import UserProfile
from padd.system import ALARM

FPS = 30.0
EAR_OPEN, EAR_CLOSED = 0.24, 0.06
RAW_PITCH_NEUTRAL, PITCH_SIGN, RAW_YAW_NEUTRAL = 5.0, -1.0, 2.0


def ear_open_at(dp, dy):
    return EAR_OPEN - 0.004 * max(dp, 0.0) + 0.001 * min(dp, 0.0) - 0.0005 * abs(dy)


def frame(t, factor=1.0, dp=0.0, dy=0.0, mar=0.03, rng=None):
    """factor: 1 = eye normally open for this pose, 0 = closed."""
    rng = rng or np.random.default_rng(0)
    ear = EAR_CLOSED + factor * (ear_open_at(dp, dy) - EAR_CLOSED) + rng.normal(0, 0.004)
    return FrameFeatures(
        t=t, ear_l=ear, ear_r=ear, ear=ear, mar=mar + rng.normal(0, 0.005),
        pitch=RAW_PITCH_NEUTRAL + PITCH_SIGN * dp + rng.normal(0, 0.3),
        yaw=RAW_YAW_NEUTRAL + dy, roll=0.0, eye_width_px=40.0,
    )


def blink_factor(t, period=3.5, dur=0.15, offset=0.7):
    return 0.0 if (t - offset) % period < dur else 1.0


def stream(seconds, fn, t0=0.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    return [frame(t0 + i / FPS, rng=rng, **fn(i / FPS)) for i in range(n)]


@pytest.fixture(scope="module")
def profile():
    targets = {"pose_center": (0, 0), "pose_left": (0, 20), "pose_right": (0, -20),
               "pose_up": (-12, 0), "pose_down": (25, 0)}

    def phase_fn(key):
        if key == "rest":
            return lambda t: {"factor": blink_factor(t)}
        if key in targets:
            dp, dy = targets[key]
            return lambda t: {"factor": blink_factor(t), "dp": dp, "dy": dy}
        if key == "talk":
            return lambda t: {"mar": 0.18 + 0.14 * math.sin(2 * math.pi * 3 * t)}
        if key == "closed":
            return lambda t: {"factor": 0.0 if t > 0.8 else 1.0}
        if key == "yawn":
            return lambda t: {"mar": 0.85 if 3 <= t <= 6.5 else 0.03}

    samples, t0 = {}, 0.0
    for ph in PHASES:
        fs = stream(ph.seconds, phase_fn(ph.key), t0=t0)
        samples[ph.key] = [(f.t - t0, f) for f in fs]
        t0 += ph.seconds
    return fit_profile(samples, "synthetic")


def run(system, frames):
    return [system.update(f, f.t) for f in frames]


def normal(t):
    return {"factor": blink_factor(t)}


# --------------------------------------------------------------------------
def test_calibration_learns_the_person(profile):
    assert profile.pitch_sign == PITCH_SIGN
    assert profile.ear_open == pytest.approx(EAR_OPEN, abs=0.01)
    assert profile.ear_closed == pytest.approx(EAR_CLOSED, abs=0.01)
    assert profile.mar_talk_p95 < profile.yawn_threshold < 0.85
    assert 12 < profile.blink_rate < 22
    # Pose model predicts the narrowed eye when looking down
    from padd.profile import PersonalModel
    m = PersonalModel(profile)
    assert m.expected_open_ear(25, 0) == pytest.approx(ear_open_at(25, 0), abs=0.015)


def test_phone_glance_fools_baseline_but_not_padd(profile):
    def fn(t):
        if 20 <= t < 30:  # quick look down at the phone, eyes open
            return {"factor": blink_factor(t), "dp": 25.0 * min(1.0, (t - 20) / 0.15)}
        return normal(t)

    frames = stream(50, fn)
    base = FixedThresholdSystem()
    padd = PADDSystem(profile)
    assert any(s.level == ALARM for s in run(base, frames))
    assert all(s.level != ALARM for s in run(padd, frames))
    assert padd.counts["nods"] == 0


def test_talking_is_not_a_yawn(profile):
    padd = PADDSystem(profile)
    run(padd, stream(30, lambda t: {"factor": blink_factor(t),
                                    "mar": 0.2 + 0.18 * math.sin(2 * math.pi * 3.5 * t)}))
    assert padd.counts["yawns"] == 0


def test_drowsy_episode_raises_alarm(profile):
    def fn(t):
        out = {"factor": 1.0}
        if t < 30:
            return normal(t)
        d = t - 30
        k = d % 3.0  # slow blinks: 0.6 s closed, 0.4 s slow reopening
        if d < 15:
            out["factor"] = 0.0 if k < 0.6 else min(1.0, (k - 0.6) / 0.4) if k < 1.0 else 1.0
        if 10 <= d < 13:
            out["mar"] = 0.85
        if 18 <= d < 20:
            out["dp"] = 20.0 * (d - 18) / 2.0  # slow droop...
        elif 20 <= d < 20.15:
            out["dp"] = 20.0 * (1 - (d - 20) / 0.15)  # ...then snap back
        if 24 <= d < 25.5:
            out["factor"] = 0.0  # long closure
        return out

    padd = PADDSystem(profile)
    statuses = run(padd, stream(60, fn))
    assert padd.counts["nods"] == 1
    assert padd.counts["yawns"] == 1
    assert padd.counts["slow_blinks"] >= 3
    assert any(s.level == ALARM for s in statuses[int(30 * FPS):])
    assert all(s.level != ALARM for s in statuses[: int(30 * FPS)])


def test_alarm_turns_off_after_recovery(profile):
    """Old evidence must not keep the alarm on once the person is alert again."""
    def drowsy(t):
        out = {"factor": 1.0}
        k = t % 3.0  # slow blinks
        out["factor"] = 0.0 if k < 0.6 else min(1.0, (k - 0.6) / 0.4) if k < 1.0 else 1.0
        for start in (5.0, 25.0):  # two yawns
            if start <= t < start + 3:
                out["mar"] = 0.85
        for start in (12.0, 32.0):  # two nods: slow droop, fast snap back
            if start <= t < start + 2:
                out["dp"] = 20.0 * (t - start) / 2.0
            elif start + 2 <= t < start + 2.15:
                out["dp"] = 20.0 * (1 - (t - start - 2) / 0.15)
        return out

    def fn(t):
        return drowsy(t) if t < 40 else normal(t)

    padd = PADDSystem(profile)
    statuses = run(padd, stream(100, fn))
    assert any(s.level == ALARM for s in statuses[: int(40 * FPS)])
    # Score is still high from the yawns and nods, but the person looks alert:
    # after a short grace period the alarm must be off (CAUTION at most).
    assert max(s.score for s in statuses[int(55 * FPS):int(60 * FPS)]) > C.ALARM_OFF
    assert all(s.level != ALARM for s in statuses[int(55 * FPS):])


def test_head_slump_raises_alarm(profile):
    """Head droops slowly and stays down (eyes open or not): falling asleep."""
    def fn(t):
        if t >= 10:
            return {"factor": blink_factor(t), "dp": 25.0 * min(1.0, (t - 10) / 3.0)}
        return normal(t)

    padd = PADDSystem(profile)
    statuses = run(padd, stream(25, fn))
    assert all(s.level != ALARM for s in statuses[: int(10 * FPS)])
    # 10 deg is reached at t = 11.2 s; slump alarm 3 s later
    first = next(i for i, s in enumerate(statuses) if s.level == ALARM) / FPS
    assert 14.0 <= first <= 15.0
    assert all(s.level == ALARM for s in statuses[int(16 * FPS):])


def test_head_down_too_long_raises_alarm_even_if_face_is_lost(profile):
    """A fast look-down is not a slump, but held for 12 s it still alarms,
    also when the face tracker loses the face with the chin on the chest."""
    def fn(t):
        if t >= 10:
            return {"factor": blink_factor(t), "dp": 25.0 * min(1.0, (t - 10) / 0.15)}
        return normal(t)

    padd = PADDSystem(profile)
    levels = []
    for f in stream(30, fn):
        face = None if f.t >= 15 else f  # face lost 5 s into the look-down
        levels.append(padd.update(face, f.t).level)
    assert all(lv != ALARM for lv in levels[: int(21.5 * FPS)])  # 10 s phone glance is fine
    assert all(lv == ALARM for lv in levels[int(22.5 * FPS):])


def test_microsleep_override(profile):
    def fn(t):
        return {"factor": 0.0} if 10 <= t < 12.5 else normal(t)

    padd = PADDSystem(profile)
    statuses = run(padd, stream(15, fn))
    first = next(i for i, s in enumerate(statuses) if s.level == ALARM) / FPS
    assert 12.0 <= first <= 12.2


def test_nod_detector_separates_nod_from_look_down():
    det = NodDetector()
    events = []
    for i in range(int(6 * FPS)):  # slow droop then snap
        t = i / FPS
        p = 0 if t < 1 else 20 * (t - 1) / 2 if t < 3 else max(0.0, 20 * (1 - (t - 3) / 0.15))
        events += [e for e in [det.update(t, p)] if e]
    assert len(events) == 1

    det = NodDetector()
    events = []
    for i in range(int(8 * FPS)):  # fast look down, hold, come back
        t = i / FPS
        p = 0 if t < 1 else 25 if t < 5 else 0
        events += [e for e in [det.update(t, p)] if e]
    assert events == []


def test_yawn_detector():
    det = YawnDetector(threshold=0.45)
    hits = [det.update(i / FPS, 0.85 if 2 <= i / FPS < 5 else 0.03) for i in range(int(8 * FPS))]
    assert sum(h is not None for h in hits) == 1

    det = YawnDetector(threshold=0.3)
    talk = [det.update(i / FPS, 0.3 + 0.2 * math.sin(2 * math.pi * 3 * i / FPS)) for i in range(int(10 * FPS))]
    assert all(h is None for h in talk)


def test_profile_roundtrip(profile, tmp_path):
    path = profile.save(tmp_path / "p.json")
    assert UserProfile.load(path) == profile


def test_scoring():
    segs = [SegmentRow(0, 10, "a", "alert", "phone"), SegmentRow(10, 20, "d", "drowsy", "drowsy_x"),
            SegmentRow(20, 40, "b", "alert", "normal")]
    times = np.arange(0, 40, 1.0)
    levels = np.zeros(40, int)
    levels[5] = ALARM  # false alarm while on the phone
    levels[14:16] = ALARM  # detection, 4 s latency
    levels[25] = ALARM  # within grace period after drowsy -> not counted
    levels[35] = ALARM  # false alarm
    r = score_run(times, levels, segs)
    assert r["false_alarms"] == 2
    assert r["fa_by_category"] == {"phone": 1, "normal": 1}
    assert r["recall"] == 1.0
    assert r["median_latency"] == 4.0
