"""PADD - Personalized Adaptive Drowsiness Detector.

Usage:
  python main.py calibrate --name alice          # ~90 s guided calibration
  python main.py run       --name alice          # live detection with PADD
  python main.py compare   --name alice          # split screen: fixed baseline vs PADD
  python main.py record    --name alice          # record a labelled test session
  python main.py evaluate  --sessions sessions   # replay sessions, write results/

Add --video path/to/file.mp4 to run/compare on a video instead of the webcam.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import cv2


def load_profile(name: str):
    from padd.profile import UserProfile, profile_path

    path = profile_path(name)
    if not path.exists():
        sys.exit(f"No profile for '{name}'. Run: python main.py calibrate --name {name}")
    return UserProfile.load(path)


def open_source(args):
    from padd.landmarks import Source

    return Source(camera=args.camera, video=args.video)


def cmd_calibrate(args) -> None:
    from padd.calibration import run_calibration
    from padd.landmarks import FaceTracker

    src = open_source(args)
    with FaceTracker() as tracker:
        profile = run_calibration(src, tracker, args.name, skip_yawn=args.skip_yawn)
    src.release()
    cv2.destroyAllWindows()
    if profile is None:
        sys.exit("Calibration cancelled.")
    path = profile.save()
    print(profile.summary())
    print(f"\nSaved to {path}")


def live_loop(args, systems, window: str) -> None:
    from padd.landmarks import FaceTracker
    from padd.ui import Beeper, draw_landmarks, draw_status, fit_to_screen, side_by_side, text

    src = open_source(args)
    beeper = Beeper(enabled=not args.no_sound)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    with FaceTracker() as tracker:
        while True:
            frame, t = src.read()
            if frame is None:
                break
            f = tracker.process(frame, t)
            panels = []
            for system in systems:
                status = system.update(f, t)
                panel = frame.copy()
                draw_landmarks(panel, f)
                draw_status(panel, system.name, status, system.alarm_count, show_cues=bool(status.cues))
                if f is None:
                    text(panel, "No face detected", (10, panel.shape[0] // 2), 0.8, (0, 210, 255), 2)
                panels.append(panel)
            beeper(status.level, status.distracted)  # sound follows the last panel (PADD)
            view = panels[0] if len(panels) == 1 else side_by_side(*panels)
            cv2.imshow(window, fit_to_screen(view))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                beeper.enabled = not beeper.enabled
            if key == ord("s"):
                shot = Path("screenshots") / f"{dt.datetime.now():%Y%m%d_%H%M%S}.png"
                shot.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(shot), view)
                print(f"Saved {shot}")
    src.release()
    cv2.destroyAllWindows()
    for system in systems:
        print(f"{system.name}: {system.alarm_count} alarm(s)")


def cmd_run(args) -> None:
    from padd.fusion import PADDSystem

    print("Keys: q = quit, m = mute, s = screenshot")
    live_loop(args, [PADDSystem(load_profile(args.name))], "PADD")


def cmd_compare(args) -> None:
    from padd.baselines import FixedThresholdSystem
    from padd.fusion import PADDSystem

    print("Keys: q = quit, m = mute, s = screenshot")
    live_loop(args, [FixedThresholdSystem(), PADDSystem(load_profile(args.name))], "PADD vs baseline")


def cmd_record(args) -> None:
    from padd.record import run_recording

    out = Path(args.out) if args.out else Path("sessions") / f"{args.name}_{dt.datetime.now():%Y%m%d_%H%M}"
    src = open_source(args)
    ok = run_recording(src, out, skip_yawn=args.skip_yawn)
    src.release()
    cv2.destroyAllWindows()
    print(f"{'Saved' if ok else 'Stopped early; partial session saved'} in {out}")


def cmd_evaluate(args) -> None:
    from padd.evaluate import run_evaluation

    run_evaluation(Path(args.sessions), Path(args.out))


def main() -> None:
    parser = argparse.ArgumentParser(description="PADD: personalized drowsiness detection")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p):
        p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
        p.add_argument("--video", help="use a video file instead of the webcam")

    p = sub.add_parser("calibrate", help="learn a personal profile (~90 s)")
    p.add_argument("--name", required=True)
    p.add_argument("--skip-yawn", action="store_true", help="skip the yawn phase")
    add_source(p)
    p.set_defaults(func=cmd_calibrate)

    for name, func, text in (("run", cmd_run, "live detection with PADD"),
                             ("compare", cmd_compare, "split screen: fixed baseline vs PADD")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--name", required=True)
        p.add_argument("--no-sound", action="store_true")
        add_source(p)
        p.set_defaults(func=func)

    p = sub.add_parser("record", help="record a labelled evaluation session")
    p.add_argument("--name", required=True, help="participant name or ID")
    p.add_argument("--out", help="output folder (default sessions/<name>_<time>)")
    p.add_argument("--skip-yawn", action="store_true")
    add_source(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("evaluate", help="replay recorded sessions and write metrics")
    p.add_argument("--sessions", default="sessions")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
