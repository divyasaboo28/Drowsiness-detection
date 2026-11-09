"""Guided personal calibration (~90 seconds) and profile fitting.

The calibration walks the user through 9 phases to learn their unique:
  - Eye Aspect Ratio range (open vs closed)
  - Blink patterns (rate, duration, reopen speed)
  - Head pose mapping (how EAR changes when looking up/down/left/right)
  - Mouth characteristics (rest, talking, yawning)

fit_profile() only needs features grouped by phase, so the same code fits
a profile live (run_calibration) and offline from a recorded session
(evaluate.py).

Author / Contributor: Praga
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from .detectors import BlinkDetector
from .features import FrameFeatures
from .profile import PersonalModel, UserProfile, pose_design
from .ui import DARK, GREEN, WHITE, YELLOW, chime, draw_landmarks, text


@dataclass(frozen=True)
class Phase:
    key: str
    seconds: float
    instruction: str
    target: str | None = None  # where to look: center/left/right/up/down
    trim: float = 1.5  # seconds ignored at the start (reaction time)


PHASES = [
    Phase("rest", 30, "Look at the screen normally and blink as you usually do", trim=2.0),
    Phase("pose_center", 5, "Look at the GREEN DOT", "center"),
    Phase("pose_left", 5, "Look at the GREEN DOT", "left"),
    Phase("pose_right", 5, "Look at the GREEN DOT", "right"),
    Phase("pose_up", 5, "Look at the GREEN DOT", "up"),
    Phase("pose_down", 6, "Look DOWN at your phone / lap, as if reading", "down"),
    Phase("talk", 15, "Talk out loud: describe your day, count, read anything", trim=1.0),
    Phase("closed", 6, "Close your eyes gently. Keep them closed until the beep"),
    Phase("yawn", 12, "Yawn widely once or twice (a fake yawn is fine)", trim=0.5),
]
POSE_PHASES = ("pose_center", "pose_left", "pose_right", "pose_up", "pose_down")

# phase key -> list of (time since phase start, features or None)
PhaseSamples = dict[str, list[tuple[float, FrameFeatures | None]]]


def phases(skip_yawn: bool = False) -> list[Phase]:
    return [p for p in PHASES if not (skip_yawn and p.key == "yawn")]


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def _valid(samples: PhaseSamples, key: str) -> list[FrameFeatures]:
    trim = next((p.trim for p in PHASES if p.key == key), 1.0)
    return [f for (tp, f) in samples.get(key, []) if f is not None and tp >= trim]


def _pct(values, q, default):
    return float(np.percentile(values, q)) if len(values) else default


def fit_profile(samples: PhaseSamples, name: str) -> UserProfile:
    """Fit a personalized UserProfile from collected phase samples.
    
    Processes the raw features recorded during each calibration phase to compute:
    1. Open and closed eye aspect ratios (with robust statistics).
    2. Head pose neutral offsets and pitch coordinate direction.
    3. Pose-normalized EAR model using regularized polynomial regression.
    4. Personal blink dynamics (duration percentiles, slow-blink rate).
    5. Baseline PERCLOS at rest.
    6. Speech vs. yawn discrimination threshold for the mouth.
    """
    warnings: list[str] = []

    for key, items in samples.items():
        if items:
            found = sum(f is not None for _, f in items) / len(items)
            if found < 0.7:
                warnings.append(f"face found in only {found:.0%} of '{key}' frames")

    # --- Phase 1: Learn normal eye metrics from the rest phase ---
    rest = _valid(samples, "rest")
    if len(rest) < 60:
        raise ValueError("Not enough face frames in the rest phase; check lighting and camera position.")
    rest_ear = np.array([f.ear for f in rest])

    # --- eye scale: open and closed ---------------------------------------
    # Use 55th percentile instead of median so normal blinks don't depress the open EAR estimate
    ear_open = _pct(rest_ear, 55, 0.3)
    # Median Absolute Deviation (MAD) scaled by 1.4826 estimates normal standard deviation robustly
    ear_open_mad = float(np.median(np.abs(rest_ear - np.median(rest_ear))) * 1.4826)
    closed = np.array([f.ear for f in _valid(samples, "closed")])
    ear_closed = float(np.median(closed)) if len(closed) >= 15 else np.nan
    if not np.isfinite(ear_closed) or ear_closed > 0.7 * ear_open:
        warnings.append("closed-eye phase unusable; using an estimated closed-eye value")
        ear_closed = 0.35 * ear_open
    ear_closed = min(ear_closed, ear_open - 0.04)

    # --- head pose ----------------------------------------------------------
    pitch_neutral = float(np.median([f.pitch for f in rest]))
    yaw_neutral = float(np.median([f.yaw for f in rest]))
    down = [f.pitch for f in _valid(samples, "pose_down")]
    up = [f.pitch for f in _valid(samples, "pose_up")]
    pitch_sign = C.DEFAULT_PITCH_SIGN
    if down and up and abs(np.median(down) - np.median(up)) >= 6:
        pitch_sign = float(np.sign(np.median(down) - np.median(up)))
    elif down and abs(np.median(down) - pitch_neutral) >= 4:
        pitch_sign = float(np.sign(np.median(down) - pitch_neutral))
    else:
        warnings.append("could not tell which way is 'head down'; using the default pitch sign")

    # --- pose-normalized eye model (N1) ----------------------------------------
    # Drop blinks per phase: looking down legitimately narrows the eye, so the
    # cut-off must be relative to that phase's own typical EAR.
    pose_frames, pts = [], []
    for key in ("rest",) + POSE_PHASES:
        fs = _valid(samples, key)
        if not fs:
            continue
        typical = float(np.median([f.ear for f in fs]))
        pts += [f for f in fs if f.ear > ear_closed + 0.6 * (typical - ear_closed)]
        if key != "rest":
            pose_frames += fs
    n_pose_keys = sum(len(_valid(samples, k)) >= 10 for k in POSE_PHASES)
    coef = [ear_open, 0.0, 0.0, 0.0]
    # Ridge regression: fit expected EAR as a function of pitch (linear + quad) and yaw
    if n_pose_keys >= 3 and len(pose_frames) >= 30:
        dp = np.array([pitch_sign * (f.pitch - pitch_neutral) for f in pts])
        dy = np.array([f.yaw - yaw_neutral for f in pts])
        X = pose_design(dp, dy)
        y = np.array([f.ear for f in pts])
        lam = 0.01 * len(y) * np.diag([0.0, 1.0, 1.0, 1.0])  # ridge: prefer "no effect"
        coef = np.linalg.solve(X.T @ X + lam, X.T @ y).tolist()
    else:
        warnings.append("pose sweep incomplete; pose normalization disabled for this profile")

    # --- Assemble the initial profile with all calibrated values ---
    profile = UserProfile(
        name=name,
        created=dt.datetime.now().isoformat(timespec="seconds"),
        ear_open=ear_open,
        ear_open_mad=ear_open_mad,
        ear_closed=ear_closed,
        pose_coef=coef,
        pitch_sign=pitch_sign,
        pitch_neutral=pitch_neutral,
        yaw_neutral=yaw_neutral,
        blink_dur_p95=C.DEFAULT_BLINK_DUR_P95,
        blink_dur_p99=C.DEFAULT_BLINK_DUR_P99,
        reopen_p95=C.DEFAULT_REOPEN_P95,
        blink_rate=0.0,
        slow_blink_rate=0.0,
        perclos_base=0.0,
        mar_rest=0.0,
        mar_talk_p95=0.0,
        yawn_threshold=C.DEFAULT_YAWN_THRESHOLD,
        warnings=warnings,
    )

    # --- personal blink signature (N2) -------------------------------------------
    model = PersonalModel(profile)
    det = BlinkDetector(slow_duration=np.inf, slow_reopen=np.inf)
    openness, events = [], []
    for f in rest:
        o = model.openness(f.ear, model.rel_pitch(f.pitch), model.rel_yaw(f.yaw))
        openness.append(o)
        events += det.update(f.t, o)
    blinks = [e for e in events if not e.long]
    minutes = max((rest[-1].t - rest[0].t) / 60.0, 1e-6)
    profile.blink_rate = len(blinks) / minutes
    profile.perclos_base = float(np.mean(np.array(openness) < C.PERCLOS_T))
    if len(blinks) >= 5:
        durs = np.array([b.duration for b in blinks])
        reopens = np.array([b.reopen for b in blinks])
        profile.blink_dur_p95 = float(np.percentile(durs, 95))
        profile.blink_dur_p99 = float(np.percentile(durs, 99))
        profile.reopen_p95 = float(np.percentile(reopens, 95))
        slow_d = max(profile.blink_dur_p95, C.MIN_SLOW_BLINK_DURATION)
        slow_r = max(profile.reopen_p95, C.MIN_SLOW_REOPEN)
        n_slow = int(np.sum((durs > slow_d) | (reopens > slow_r)))
        profile.slow_blink_rate = n_slow / minutes
    else:
        warnings.append(f"only {len(blinks)} blinks seen at rest; using default blink thresholds")

    # --- mouth ---------------------------------------------------------------
    profile.mar_rest = float(np.median([f.mar for f in rest]))
    talk = [f.mar for f in _valid(samples, "talk")]
    if len(talk) >= 30:
        profile.mar_talk_p95 = _pct(talk, 95, 0.0)
    else:
        profile.mar_talk_p95 = profile.mar_rest + 0.2
        warnings.append("no talking data; yawn threshold may confuse speech with yawns")
    lo = max(profile.mar_talk_p95 * 1.1, C.YAWN_MIN_THRESHOLD)
    yawn = [f.mar for f in _valid(samples, "yawn")]
    if len(yawn) >= 30:
        peak = _pct(yawn, 95, 0.0)
        profile.yawn_threshold = max(lo, profile.mar_rest + 0.6 * (peak - profile.mar_rest))
        if profile.yawn_threshold > 0.9 * peak:
            warnings.append("your yawn was not much wider than talking; yawn detection may be weak")
    else:
        profile.yawn_threshold = max(lo, C.DEFAULT_YAWN_THRESHOLD)
    return profile


# --------------------------------------------------------------------------
# Live, guided calibration
# --------------------------------------------------------------------------
def target_point(target: str | None, w: int, h: int) -> tuple[int, int] | None:
    m = 40
    return {
        "center": (w // 2, h // 2),
        "left": (m, h // 2),
        "right": (w - m, h // 2),
        "up": (w // 2, m),
    }.get(target or "")


def draw_phase(img, phase: Phase, t_in: float, step: int, n_steps: int) -> None:
    h, w = img.shape[:2]
    remaining = max(0.0, phase.seconds - t_in)
    cv2.rectangle(img, (0, h - 70), (w, h), DARK, -1)
    text(img, f"Calibration {step}/{n_steps}: {phase.instruction}", (15, h - 40), 0.65, WHITE, 2, bg=False)
    prog = int(w * min(1.0, t_in / phase.seconds))
    cv2.rectangle(img, (0, h - 12), (prog, h), GREEN, -1)
    text(img, f"{remaining:4.1f}s", (w - 80, h - 40), 0.65, YELLOW, 2, bg=False)
    pt = target_point(phase.target, w, h)
    if pt:
        cv2.circle(img, pt, 22, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 6, DARK, -1, cv2.LINE_AA)
    if phase.target == "down":
        cv2.arrowedLine(img, (w // 2, h // 2), (w // 2, h - 100), GREEN, 8, cv2.LINE_AA, tipLength=0.25)


def run_calibration(source, tracker, name: str, skip_yawn: bool = False,
                    window: str = "PADD calibration") -> UserProfile | None:
    """Run interactive 9-phase calibration via GUI and return the fitted UserProfile."""
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    plan = phases(skip_yawn)
    samples: PhaseSamples = {p.key: [] for p in plan}

    # Wait for the user to be ready: show the camera feed and check for a face.
    # Start the calibration phases only after a face has been visible for 3 seconds.
    t_ready = None
    while True:
        frame, t = source.read()
        if frame is None:
            return None
        f = tracker.process(frame, t)
        t_ready = t if t_ready is None else t_ready
        draw_landmarks(frame, f)
        msg = "Face found - starting soon" if f else "No face found: sit in front of the camera"
        text(frame, f"PADD calibration for '{name}'. {msg}", (15, 40), 0.7, WHITE, 2)
        text(frame, "Follow the instructions at the bottom. ESC to cancel.", (15, 75), 0.6)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow(window)
            return None
        if f is None:
            t_ready = t
        elif t - t_ready > 3.0:
            break

    for step, phase in enumerate(plan, 1):
        chime()
        t_start = None
        while True:
            frame, t = source.read()
            if frame is None:
                break
            t_start = t if t_start is None else t_start
            t_in = t - t_start
            if t_in >= phase.seconds:
                break
            f = tracker.process(frame, t)
            samples[phase.key].append((t_in, f))
            draw_landmarks(frame, f)
            draw_phase(frame, phase, t_in, step, len(plan))
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(window)
                return None
    chime()
    cv2.destroyWindow(window)
    return fit_profile(samples, name)
