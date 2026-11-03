"""Video input and the MediaPipe Face Landmarker wrapper."""

from __future__ import annotations

import csv
import os
import time
import urllib.request
from pathlib import Path

import cv2

from . import config as C
from .features import FrameFeatures, from_mediapipe


def ensure_model(path: str | Path = C.MODEL_PATH) -> Path:
    """Download the Face Landmarker model (~4 MB) on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(C.MODEL_URL, tmp)
        tmp.replace(path)
    return path


class FaceTracker:
    def __init__(self, model_path: str | Path = C.MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr, t: float) -> FrameFeatures | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)  # MediaPipe needs increasing timestamps
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        return from_mediapipe(result, w, h, t)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Source:
    """Webcam (mirrored, wall-clock time) or video file (file time).

    For recorded sessions, timestamps come from frames.csv next to the video
    so replays see exactly the timing of the live recording.
    """

    def __init__(self, camera: int | None = 0, video: str | None = None,
                 width: int = 1280, height: int = 720):
        self.is_file = video is not None
        self.times: list[float] | None = None
        if self.is_file:
            self.cap = cv2.VideoCapture(str(video))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames_csv = Path(video).with_name("frames.csv")
            if frames_csv.exists():
                with open(frames_csv, newline="") as fh:
                    self.times = [float(r["t"]) for r in csv.DictReader(fh)]
        else:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.fps = 30.0
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {'video ' + str(video) if video else f'camera {camera}'}")
        self.index = 0
        self.t0 = time.monotonic()

    def read(self):
        """Return (frame, t) or (None, None) at the end of a file."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_file:
            if self.times is not None and self.index < len(self.times):
                t = self.times[self.index]
            else:
                t = self.index / self.fps
        else:
            frame = cv2.flip(frame, 1)  # mirror view feels natural
            t = time.monotonic() - self.t0
        self.index += 1
        return frame, t

    def release(self) -> None:
        self.cap.release()
