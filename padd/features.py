"""Per-frame geometric features extracted from MediaPipe face landmarks.

This module computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and head pose (pitch/yaw/roll) from the 478-point face mesh. These
features are the raw inputs to calibration and all detection systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C

# Fields stored when features are cached to disk (see evaluate.py)
CACHE_FIELDS = (
    "ear_l", "ear_r", "ear", "mar", "pitch", "yaw", "roll",
    "eye_width_px", "blink_bs", "jaw_bs",
)


@dataclass
class FrameFeatures:
    """Container for all per-frame measurements extracted from one video frame.
    
    EAR (Eye Aspect Ratio): ratio of eye height to width — drops when eyes close.
    MAR (Mouth Aspect Ratio): ratio of mouth height to width — rises during yawns.
    Pitch/Yaw/Roll: head orientation in degrees from MediaPipe's face transform.
    """
    t: float
    ear_l: float
    ear_r: float
    ear: float  # quality-weighted average of both eyes
    mar: float
    pitch: float  # raw degrees from MediaPipe; sign is fixed by calibration
    yaw: float
    roll: float
    eye_width_px: float
    blink_bs: float = 0.0  # MediaPipe blendshape eyeBlink (avg of both eyes)
    jaw_bs: float = 0.0  # MediaPipe blendshape jawOpen
    points: np.ndarray | None = field(default=None, repr=False)  # (N, 2) pixels


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> tuple[float, float]:
    """Return (EAR, eye width in px) for one eye given landmark indices p1..p6."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0, 0.0
    # EAR formula: average of two vertical distances divided by the horizontal distance.
    # EAR ≈ 0.3 when open, drops toward 0 when the eye closes.
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width), width


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    width = _dist(pts[C.MOUTH_CORNERS[0]], pts[C.MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    # Average the three vertical mouth distances and normalize by mouth width.
    # MAR rises significantly during yawns (>0.5) vs normal speech (<0.3).
    vertical = sum(_dist(pts[a], pts[b]) for a, b in C.MOUTH_VERTICAL)
    return vertical / (len(C.MOUTH_VERTICAL) * width)


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's 4x4 face transform."""
    r = np.array(m, dtype=float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0, keepdims=True)  # strip any scale
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def from_points(
    pts: np.ndarray,
    t: float,
    pose: tuple[float, float, float],
    blendshapes: dict[str, float] | None = None,
) -> FrameFeatures:
    ear_l, w_l = eye_aspect_ratio(pts, C.LEFT_EYE)
    ear_r, w_r = eye_aspect_ratio(pts, C.RIGHT_EYE)
    # When the head turns, the far eye is foreshortened and less reliable,
    # so weight each eye by its apparent width.
    ear = (ear_l * w_l + ear_r * w_r) / (w_l + w_r) if (w_l + w_r) > 0 else 0.0
    # Optionally include MediaPipe blendshape scores (eyeBlink, jawOpen)
    # for the blendshape baseline detector.
    bs = blendshapes or {}
    return FrameFeatures(
        t=t,
        ear_l=ear_l,
        ear_r=ear_r,
        ear=ear,
        mar=mouth_aspect_ratio(pts),
        pitch=pose[0],
        yaw=pose[1],
        roll=pose[2],
        eye_width_px=max(w_l, w_r),
        blink_bs=0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)),
        jaw_bs=bs.get("jawOpen", 0.0),
        points=pts,
    )


def from_mediapipe(result, width: int, height: int, t: float) -> FrameFeatures | None:
    """Convert a FaceLandmarkerResult into FrameFeatures (None if no face)."""
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    # Convert normalized [0,1] landmark coordinates to pixel coordinates
    pts = np.array([(lm.x * width, lm.y * height) for lm in lms], dtype=np.float32)
    if result.facial_transformation_matrixes:
        pose = euler_from_matrix(result.facial_transformation_matrixes[0])
    else:
        pose = (0.0, 0.0, 0.0)
    blend = None
    if result.face_blendshapes:
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
    return from_points(pts, t, pose, blend)


def yaw_confidence(rel_yaw: float) -> float:
    """1.0 while facing the camera, fading to 0 as the head turns away."""
    a = abs(rel_yaw)
    if a <= C.YAW_FULL_CONF:
        return 1.0
    return float(np.clip(1.0 - (a - C.YAW_FULL_CONF) / (C.YAW_ZERO_CONF - C.YAW_FULL_CONF), 0.0, 1.0))


def size_confidence(eye_width_px: float) -> float:
    """Tiny faces (far from the camera) give noisy landmarks."""
    return float(np.clip(eye_width_px / (2.0 * C.MIN_EYE_WIDTH_PX), 0.0, 1.0))
