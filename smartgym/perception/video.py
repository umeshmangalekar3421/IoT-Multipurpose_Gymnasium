"""Video I/O: frame iteration with real timestamps, seeking and sampling.

Written around ``cv2.VideoCapture`` because that is the lowest common
denominator between a laptop, a Jetson and an Android build (OpenCV is already
there), but the rest of the pipeline only depends on :class:`VideoSource`, so a
GStreamer/RTSP or CSI-camera source can be dropped in for live matches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_s: float
    fourcc: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
            "duration_s": round(self.duration_s, 2),
            "fourcc": self.fourcc,
        }


class VideoSource:
    """Readable video with sane fps / frame-count fallbacks.

    Many phone clips report ``fps=0`` or ``n_frames=0``; we fall back to a
    configurable default so timestamps never collapse to zero (which would
    break every velocity computation downstream).
    """

    def __init__(self, path: str, default_fps: float = 30.0,
                 max_frames: Optional[int] = None, stride: int = 1):
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv-python is required for video input")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video {path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else float(default_fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.max_frames:
            reported = min(reported, self.max_frames) if reported else self.max_frames
        self.n_frames = max(0, reported)
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0)
        self.fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_int else ""

    # -- metadata -----------------------------------------------------------
    @property
    def meta(self) -> VideoMeta:
        dur = self.n_frames / self.fps if self.fps else 0.0
        return VideoMeta(self.path, self.fps, self.width, self.height,
                         self.n_frames, dur, self.fourcc)

    def time_of(self, frame: int) -> float:
        return frame / self.fps if self.fps else 0.0

    def frame_of(self, t: float) -> int:
        return int(round(t * self.fps))

    # -- reading ------------------------------------------------------------
    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp_s, bgr_frame)`` honouring stride."""
        idx = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % self.stride == 0:
                yield idx, self.time_of(idx), frame
            idx += 1
            if self.max_frames and idx >= self.max_frames:
                break

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Random access (used by the evidence pass)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_window(self, start: int, end: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Sequential read of ``[start, end)`` — much faster than random seeks."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        idx = int(start)
        while idx < end:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def grab_first_frame(self) -> Optional[np.ndarray]:
        return self.read_frame(0)

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def read_calibration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_calibration(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
