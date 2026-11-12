"""OpenCV overlays, the split-screen comparison view, and the alarm beep.

Provides all the visual feedback for the live detection window:
  - Face landmark drawing (eye points in cyan, mouth lines in pink)
  - Status banner with color-coded alert level (green/yellow/red)
  - Orange "eyes on the road" banner for the distraction warning
  - Per-cue progress bars and fused drowsiness score bar
  - Alarm reason text at the bottom of the frame
  - Non-blocking audio alerts using the system beeper
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from . import config as C
from .features import FrameFeatures
from .system import ALARM, CAUTION, LEVEL_NAMES, Status

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 200, 80)
YELLOW = (0, 210, 255)
RED = (40, 40, 230)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
DARK = (25, 25, 25)
LEVEL_COLORS = {0: GREEN, CAUTION: YELLOW, ALARM: RED}
CUE_LABELS = {"eye": "Eyes", "blink": "Slow blinks", "yawn": "Yawns", "nod": "Head nods"}


def text(img, s, org, scale=0.55, color=WHITE, thick=1, bg=True):
    """Draw anti-aliased text with an optional dark rectangle background."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if bg:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), DARK, -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_landmarks(img, f: FrameFeatures | None) -> None:
    """Draw the eye landmarks (cyan dots) and mouth lines (pink) onto the image."""
    if f is None or f.points is None:
        return
    pts = f.points.astype(int)
    for idx in (C.LEFT_EYE, C.RIGHT_EYE):
        for i in idx:
            cv2.circle(img, tuple(pts[i]), 2, (255, 220, 0), -1, cv2.LINE_AA)
    for a, b in C.MOUTH_VERTICAL:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 120, 255), 1, cv2.LINE_AA)
    cv2.line(img, tuple(pts[C.MOUTH_CORNERS[0]]), tuple(pts[C.MOUTH_CORNERS[1]]),
             (255, 120, 255), 1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, color, label=None, vmax=1.0):
    """Draw a horizontal progress/status bar with background and optional label."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * float(np.clip(value / vmax, 0, 1)))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    if label:
        text(img, label, (x, y - 4), 0.45, WHITE, bg=False)


def draw_status(img, title: str, status: Status, alarms: int, show_cues: bool) -> None:
    """Draw full PADD status overlay: top banner, metrics, cue bars, and reason string."""
    h, w = img.shape[:2]
    color = LEVEL_COLORS[status.level]
    # Banner
    cv2.rectangle(img, (0, 0), (w, 34), color, -1)
    cv2.putText(img, f"{title}  |  {LEVEL_NAMES[status.level]}", (10, 23), FONT, 0.6, DARK, 2, cv2.LINE_AA)
    if status.level == ALARM:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 8)
    elif status.distracted:
        cv2.rectangle(img, (0, h - 68), (w, h - 38), ORANGE, -1)
        cv2.putText(img, "EYES ON THE ROAD", (10, h - 46), FONT, 0.7, DARK, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), ORANGE, 4)

    y = 60
    info = status.info
    if "ear" in info:
        ref = info.get("ear_ref")
        text(img, f"EAR {info['ear']:.3f}" + (f"  (ref {ref:.3f})" if ref is not None else ""), (10, y))
        y += 24
    if "openness" in info:
        text(img, f"Openness {info['openness'] * 100:5.0f}%", (10, y))
        y += 24
    if "mar" in info:
        text(img, f"MAR {info['mar']:.2f}", (10, y))
        y += 24
    if "pitch" in info:
        text(img, f"Head pitch {info['pitch']:+.0f} deg", (10, y))
        y += 24
    if "perclos" in info:
        text(img, f"PERCLOS {info['perclos'] * 100:.0f}%", (10, y))
        y += 24
    if "blink_bs" in info:
        text(img, f"eyeBlink {info['blink_bs']:.2f}  jawOpen {info['jaw_bs']:.2f}", (10, y))
        y += 24
    text(img, f"Alarms: {alarms}", (10, y), 0.55, YELLOW)

    if show_cues and status.cues:
        x0, y0, bw = w - 200, 60, 180
        for i, (k, v) in enumerate(status.cues.items()):
            draw_bar(img, x0, y0 + i * 38 + 14, bw, 10, v, (255, 180, 80), CUE_LABELS[k], C.WEIGHTS[k])
        yb = y0 + len(status.cues) * 38 + 14
        draw_bar(img, x0, yb, bw, 14, status.score, color, "Drowsiness score")
        for thr in (C.CAUTION_ON, C.ALARM_ON):
            xt = x0 + int(bw * thr)
            cv2.line(img, (xt, yb - 2), (xt, yb + 16), WHITE, 1)

    if status.reasons:
        msg = "Why: " + ", ".join(status.reasons)
        text(img, msg, (10, h - 16), 0.55, color)


def side_by_side(left, right):
    """Stack two camera panels side by side with a vertical separator bar."""
    sep = np.full((left.shape[0], 6, 3), 40, np.uint8)
    return np.hstack([left, sep, right])


def fit_to_screen(img, max_w=1600):
    """Downscale image if wider than max_w so it fits nicely on screen."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (int(w * s), int(h * s)))


def chime() -> None:
    """Short non-blocking tone for calibration / recording phase changes."""
    def play():
        if os.name == "nt":
            import winsound
            winsound.Beep(880, 150)
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=play, daemon=True).start()


class Beeper:
    """Non-blocking, rate-limited alarm sound player.
    
    Plays a loud warning beep while the system is in ALARM state (at most
    once per BEEP_INTERVAL, 0.8s), and a short, soft beep for the
    "eyes on the road" distraction warning (at most once per
    DISTRACT_BEEP_INTERVAL, 1.5s). Runs the sound in a background thread
    so it never blocks the video processing loop.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last = 0.0

    def __call__(self, level: int, distracted: bool = False) -> None:
        """Loud beep for a drowsiness ALARM; a short, soft beep for distraction."""
        if not self.enabled:
            return
        if level == ALARM:
            sound, interval = (C.BEEP_FREQ, C.BEEP_MS), C.BEEP_INTERVAL
        elif distracted:
            sound, interval = (C.DISTRACT_BEEP_FREQ, C.DISTRACT_BEEP_MS), C.DISTRACT_BEEP_INTERVAL
        else:
            return
        now = time.monotonic()
        if now - self._last < interval:
            return
        self._last = now
        threading.Thread(target=self._play, args=sound, daemon=True).start()

    @staticmethod
    def _play(freq: int, ms: int) -> None:
        if os.name == "nt":
            import winsound
            winsound.Beep(freq, ms)
        else:
            print("\a", end="", flush=True)
