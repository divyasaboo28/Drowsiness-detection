"""Record a labelled evaluation session: calibration + a scripted test drive.

The recording has two parts:
  1. Full calibration phases (same as `main.py calibrate`)
  2. A scripted scenario mixing normal driving behaviors and acted drowsy
     episodes, with precise timing labels for later evaluation.

Output folder contains:
  video.mp4      Raw (mirrored) webcam frames with no overlays
  frames.csv     Frame index -> timestamp mapping for faithful replay timing  
  segments.csv   Ground truth: start, end, phase, label (alert/drowsy), category

The script includes common false-alarm triggers (phone use, talking,
mirror checks, laughing) alongside drowsy episodes (slow blinks, head
nods, microsleeps) so evaluation can measure both recall and specificity.

Author / Contributor: Praga
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2

from .calibration import draw_phase, phases
from .ui import DARK, RED, WHITE, YELLOW, chime, text


@dataclass(frozen=True)
class Segment:
    """One step in the recording script with timing and ground truth label.
    
    Attributes:
        category:    Scenario type (normal, phone, talk, drowsy_blinks, etc.)
        label:       Ground truth — "alert" or "drowsy"
        seconds:     How long this segment lasts
        instruction: On-screen text telling the participant what to do
    """
    category: str
    label: str  # "alert" or "drowsy"
    seconds: float
    instruction: str


SCRIPT = [
    Segment("normal", "alert", 45, "Act normal: look at the screen, read, think"),
    Segment("phone", "alert", 10, "Look DOWN at your phone and read something on it"),
    Segment("normal", "alert", 20, "Act normal"),
    Segment("drowsy_blinks", "drowsy", 25, "DROWSY: slow, heavy blinks, eyes half-closed, one yawn"),
    Segment("normal", "alert", 30, "Act normal (wake up!)"),
    Segment("talk", "alert", 30, "Talk out loud, animatedly, about anything"),
    Segment("phone", "alert", 10, "Look DOWN at your phone again"),
    Segment("mirror", "alert", 12, "Check 'mirrors': turn your head left and right a few times"),
    Segment("drowsy_nod", "drowsy", 20, "DROWSY: let your head slowly droop forward, then jerk it up"),
    Segment("normal", "alert", 30, "Act normal"),
    Segment("laugh", "alert", 10, "Laugh / smile broadly"),
    Segment("drowsy_microsleep", "drowsy", 10, "DROWSY: close your eyes for about 3 seconds, then open"),
    Segment("normal", "alert", 30, "Act normal"),
    Segment("phone", "alert", 10, "Look DOWN at your phone"),
    Segment("drowsy_combo", "drowsy", 35, "DROWSY: slow blinks, yawns, head droops - the full act"),
    Segment("normal", "alert", 30, "Act normal"),
]


def draw_segment(img, seg: Segment, t_in: float, step: int, n: int) -> None:
    h, w = img.shape[:2]
    color = RED if seg.label == "drowsy" else (80, 200, 80)
    cv2.rectangle(img, (0, h - 70), (w, h), DARK, -1)
    text(img, f"Test {step}/{n}: {seg.instruction}", (15, h - 40), 0.65, WHITE, 2, bg=False)
    cv2.rectangle(img, (0, h - 12), (int(w * min(1.0, t_in / seg.seconds)), h), color, -1)
    text(img, f"{max(0.0, seg.seconds - t_in):4.1f}s", (w - 80, h - 40), 0.65, YELLOW, 2, bg=False)
    text(img, "REC", (w - 70, 35), 0.7, RED, 2)


def run_recording(source, out_dir: Path, skip_yawn: bool = False,
                  window: str = "PADD recording") -> bool:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    writer = None
    frame_times: list[float] = []
    rows: list[dict] = []
    calib = phases(skip_yawn)
    # Build the full recording plan: calibration phases first, then the test script
    steps = [("calib", p) for p in calib] + [("script", s) for s in SCRIPT]
    t_origin = None
    aborted = False

    for i, (kind, item) in enumerate(steps):
        chime()
        t_start = None
        while True:
            frame, t = source.read()
            if frame is None:
                aborted = True
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "video.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                t_origin = t
            t_rel = t - t_origin
            t_start = t_rel if t_start is None else t_start
            t_in = t_rel - t_start
            if t_in >= item.seconds:
                break
            # Save the clean raw frame BEFORE drawing any overlays on it
            writer.write(frame)
            frame_times.append(t_rel)

            shown = frame.copy()
            if kind == "calib":
                draw_phase(shown, item, t_in, i + 1, len(calib))
            else:
                draw_segment(shown, item, t_in, i + 1 - len(calib), len(SCRIPT))
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF == 27:
                aborted = True
                break
        if t_start is not None:
            end = frame_times[-1] if frame_times else t_start
            if kind == "calib":
                rows.append(dict(start=t_start, end=end, phase=item.key, label="calib", category=item.key))
            else:
                rows.append(dict(start=t_start, end=end, phase=f"s{i:02d}_{item.category}",
                                 label=item.label, category=item.category))
        if aborted:
            break

    chime()
    if writer is not None:
        writer.release()
    cv2.destroyWindow(window)
    with open(out_dir / "frames.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "t"])
        wr.writerows((k, f"{tt:.4f}") for k, tt in enumerate(frame_times))
    with open(out_dir / "segments.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["start", "end", "phase", "label", "category"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, "start": f"{r['start']:.4f}", "end": f"{r['end']:.4f}"})
    return not aborted
