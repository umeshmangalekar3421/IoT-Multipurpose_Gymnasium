"""Ball / shuttle detection.

Two interchangeable detectors with the same interface:

``ClassicalBallDetector``
    Motion + appearance fusion.  Works out of the box on phone video, needs no
    training data and is fast enough for real-time on a CPU.  It exploits two
    facts: a shuttlecock is the brightest, fastest small object on the court,
    and a tennis ball has a very specific yellow-green hue.

``YoloBallDetector``
    Adapter for a trained YOLO/ONNX model (``ultralytics``).  Use it once you
    have labelled a few hundred frames — see ``docs/TRAINING.md``.

Both return :class:`~smartgym.core.types.Detection` objects in image pixels and
accept a *hint* (the tracker's prediction) so the search can be restricted to a
small ROI.  That ROI hint is what makes the classical detector reliable: full
frame scanning finds spectators, shoes and line reflections, a 120 px window
around the prediction rarely does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Detection, Point

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


PROFILES: dict = {
    "shuttle": {
        "hsv_low": (0, 0, 205),
        "hsv_high": (180, 45, 255),      # white, low saturation, *very* bright
        "area": (6.0, 900.0),
        "motion_weight": 0.65,
        "color_weight": 0.35,
        "diff_thresh": 22,
    },
    "tennis_ball": {
        "hsv_low": (22, 70, 70),
        "hsv_high": (75, 255, 255),      # yellow-green
        "area": (12.0, 1600.0),
        "motion_weight": 0.35,
        "color_weight": 0.65,
        "diff_thresh": 16,
    },
    "generic": {
        "hsv_low": (0, 0, 150),
        "hsv_high": (180, 255, 255),
        "area": (8.0, 2000.0),
        "motion_weight": 0.55,
        "color_weight": 0.45,
        "diff_thresh": 20,
    },
}


class BaseBallDetector:
    name = "base"

    def initialize(self, frames: Sequence[np.ndarray]) -> None:
        """Optional pre-pass over a few frames (default: no-op)."""

    def detect(self, frame: np.ndarray,
               hint: Optional[Point] = None) -> List[Detection]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class ClassicalBallDetector(BaseBallDetector):
    """Motion + colour blob detector with a tracker-guided ROI."""

    name = "classical"

    def __init__(self, profile: str = "shuttle", roi_px: int = 130,
                 max_candidates: int = 5, min_conf: float = 0.45,
                 search_full_frame_every: int = 30, max_aspect: float = 4.5,
                 hot_decay: float = 0.90, hot_threshold: float = 7.0):
        self.max_aspect = float(max_aspect)
        self.hot_decay = float(hot_decay)
        self.hot_threshold = float(hot_threshold)
        self._hot: Optional[np.ndarray] = None
        self.p = dict(PROFILES.get(profile, PROFILES["generic"]))
        self.roi_px = int(roi_px)
        self.max_candidates = int(max_candidates)
        self.min_conf = float(min_conf)
        self.search_full_frame_every = int(search_full_frame_every)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev2_gray: Optional[np.ndarray] = None
        self._frame_idx = 0
        self._bg_gray: Optional[np.ndarray] = None

    # -- helpers ------------------------------------------------------------
    def initialize(self, frames: Sequence[np.ndarray]) -> None:
        """Build a static background model from a sample of the clip.

        With a background estimate, "motion" becomes "differs from the empty
        court", which is dramatically more reliable than frame differencing when
        the ball moves only a few pixels between frames (60 fps footage, distant
        camera).  Players are still rejected downstream by the large-blob
        suppression and the isolation test.
        """
        if cv2 is None:  # pragma: no cover
            return
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames if f is not None]
        if len(grays) < 3:
            return
        h, w = grays[0].shape[:2]
        grays = [g if g.shape[:2] == (h, w) else cv2.resize(g, (w, h)) for g in grays]
        self._bg_gray = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)

    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        """Motion mask with large moving objects (players) suppressed.

        Players are the dominant *moving* thing on a court, and their heads,
        hands and shoes produce compact blobs that look exactly like a ball.
        A large-kernel opening isolates those big regions; subtracting them
        leaves the fast, small, isolated object we actually want.  This single
        step removes most tracking failures on real footage.
        """
        if self._bg_gray is not None and self._bg_gray.shape == gray.shape:
            bg_diff = cv2.absdiff(gray, self._bg_gray)
            _, bg_mask = cv2.threshold(bg_diff, self.p["diff_thresh"], 255,
                                       cv2.THRESH_BINARY)
            # combine with inter-frame motion: the background model catches slow
            # balls, frame differencing catches anything the model has baked in
            if self._prev_gray is not None:
                inter = cv2.absdiff(gray, self._prev_gray)
                _, inter_mask = cv2.threshold(inter, self.p["diff_thresh"], 255,
                                              cv2.THRESH_BINARY)
                bg_mask = cv2.bitwise_and(bg_mask, cv2.max(inter_mask, cv2.erode(
                    bg_mask, np.ones((3, 3), np.uint8))))
            mask = bg_mask
            large_k = max(9, int(0.02 * gray.shape[1]) | 1)
            large = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                     np.ones((large_k, large_k), np.uint8))
            large = cv2.dilate(large, np.ones((large_k, large_k), np.uint8))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(large))
            return cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                    np.ones((3, 3), np.uint8))

        if self._prev_gray is None:
            return np.zeros_like(gray)
        d1 = cv2.absdiff(gray, self._prev_gray)
        d2 = (cv2.absdiff(self._prev_gray, self._prev2_gray)
              if self._prev2_gray is not None else d1)
        diff = cv2.max(d1, d2)
        _, mask = cv2.threshold(diff, self.p["diff_thresh"], 255, cv2.THRESH_BINARY)

        big_k = max(9, int(0.02 * gray.shape[1]) | 1)
        large = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                 np.ones((big_k, big_k), np.uint8))
        large = cv2.dilate(large, np.ones((big_k, big_k), np.uint8))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(large))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        return mask

    def _contrast(self, gray: np.ndarray, x: int, y: int, r: int = 9) -> float:
        """Local contrast of a blob: how much it stands out from its surround."""
        h, w = gray.shape[:2]
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 9:
            return 0.0
        cy, cx = y - y0, x - x0
        inner = patch[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
        if inner.size == 0:
            return 0.0
        return float(inner.max()) - float(patch.mean())

    def _color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(self.p["hsv_low"]), np.array(self.p["hsv_high"]))

    @staticmethod
    def _roi(frame: np.ndarray, hint: Optional[Point], roi_px: int
             ) -> Tuple[np.ndarray, int, int]:
        if hint is None:
            return frame, 0, 0
        h, w = frame.shape[:2]
        half = roi_px // 2
        x0 = int(np.clip(hint[0] - half, 0, max(0, w - 1)))
        y0 = int(np.clip(hint[1] - half, 0, max(0, h - 1)))
        x1 = int(np.clip(hint[0] + half, x0 + 2, w))
        y1 = int(np.clip(hint[1] + half, y0 + 2, h))
        return frame[y0:y1, x0:x1], x0, y0

    # -- main ---------------------------------------------------------------
    def detect(self, frame: np.ndarray, hint: Optional[Point] = None) -> List[Detection]:
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv required for the classical detector")
        self._frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full = hint is None or (self._frame_idx % self.search_full_frame_every == 0)
        roi, ox, oy = (frame, 0, 0) if full else self._roi(frame, hint, self.roi_px)

        gray_roi = gray[oy:oy + roi.shape[0], ox:ox + roi.shape[1]]
        motion = self._motion_mask(gray)
        motion_roi = motion[oy:oy + roi.shape[0], ox:ox + roi.shape[1]]
        color_roi = self._color_mask(roi)

        score = (self.p["motion_weight"] * (motion_roi > 0).astype(np.float32)
                 + self.p["color_weight"] * (color_roi > 0).astype(np.float32))
        mask = (score > 0.25).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        a_min, a_max = self.p["area"]
        dets: List[Detection] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < a_min or area > a_max:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if aspect > 6.0 or aspect < 1 / 6.0:
                continue
            # compact blobs score better; long streaks are usually motion blur
            fill = area / max(w * h, 1)
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [c], -1, 255, -1)
            overlap = float((score[blob > 0]).mean()) if np.any(blob) else 0.0
            contrast = self._contrast(gray_roi, int(cx), int(cy))
            if contrast < 18.0:
                continue
            if not self._isolated(motion_roi, int(cx), int(cy)):
                continue
            conf = float(np.clip(0.35 * min(fill * 2, 1.0) + 0.45 * overlap
                                 + 0.20 * min(contrast / 120.0, 1.0), 0, 1))
            if hint is not None:
                dist = float(np.hypot(ox + cx - hint[0], oy + cy - hint[1]))
                conf *= float(np.clip(1.25 - dist / (self.roi_px * 1.6), 0.05, 1.0))
            else:
                conf *= 0.8      # full-frame scan is less trustworthy
            if conf < self.min_conf:
                continue
            dets.append(Detection(
                xy=(float(ox + cx), float(oy + cy)),
                conf=conf, label="ball",
                box=(float(ox + x), float(oy + y), float(ox + x + w), float(oy + y + h)),
            ))

        # --- peak candidates -------------------------------------------
        # A fast ball leaves a motion streak that merges with painted lines,
        # so the contour above can exceed the area limit.  The response map
        # "moving AND ball-coloured" is line-free (lines do not move), and its
        # local maxima localise the ball accurately even inside a streak.
        peaks = self._peak_candidates(gray_roi, motion_roi, color_roi, ox, oy, hint)
        dets.extend(peaks)

        # merge near-duplicates, keep the most confident
        dets.sort(key=lambda d: -d.conf)
        merged: List[Detection] = []
        for d in dets:
            if all(np.hypot(d.xy[0] - m.xy[0], d.xy[1] - m.xy[1]) > 8.0 for m in merged):
                merged.append(d)
        dets = merged[: self.max_candidates]

        # --- persistent-detection ("hot cell") rejection -----------------
        # Anything that keeps being detected in the same place is part of the
        # scene, not a ball: flickering painted lines, a bright spectator, a
        # reflection.  A decaying occupancy grid over the frame learns this
        # online, per camera, with no configuration.
        dets = self._reject_hot(dets, frame.shape[:2])

        self._prev2_gray = self._prev_gray
        self._prev_gray = gray
        return dets

    def _isolated(self, motion_roi: np.ndarray, x: int, y: int,
                  radius: int = 26, max_fraction: float = 0.42) -> bool:
        """Reject blobs that sit inside a large moving region (a player).

        A ball is a small object surrounded by static court; a head, hand, shoe
        or racket is attached to a big moving mass.  Measuring the fraction of
        motion pixels in a disc around the candidate separates the two without
        needing a trained detector.
        """
        h, w = motion_roi.shape[:2]
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        patch = motion_roi[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        disc = (xx * xx + yy * yy) <= radius * radius
        if disc.sum() == 0:
            return False
        frac = float((patch[disc] > 0).mean())
        return frac <= max_fraction

    def _reject_hot(self, dets: List[Detection], shape: Tuple[int, int]) -> List[Detection]:
        h, w = shape
        rows, cols = 24, 32
        if self._hot is None or self._hot.shape != (rows, cols):
            self._hot = np.zeros((rows, cols), dtype=np.float32)
        self._hot *= self.hot_decay
        kept: List[Detection] = []
        for d in dets:
            r = int(np.clip(d.xy[1] / max(1, h) * rows, 0, rows - 1))
            c = int(np.clip(d.xy[0] / max(1, w) * cols, 0, cols - 1))
            if self._hot[r, c] >= self.hot_threshold:
                continue
            kept.append(d)
        for d in kept:
            r = int(np.clip(d.xy[1] / max(1, h) * rows, 0, rows - 1))
            c = int(np.clip(d.xy[0] / max(1, w) * cols, 0, cols - 1))
            self._hot[r, c] += 1.0
        return kept

    def _peak_candidates(self, gray_roi: np.ndarray, motion_roi: np.ndarray,
                         color_roi: np.ndarray, ox: int, oy: int,
                         hint: Optional[Point]) -> List[Detection]:
        if gray_roi.size == 0:
            return []
        resp = ((motion_roi > 0) & (color_roi > 0)).astype(np.float32)
        if resp.max() <= 0:
            return []
        # kill single-pixel noise (a specular speck on a player's head is not a
        # ball: a real ball occupies several pixels even when motion-blurred)
        resp = cv2.morphologyEx(resp, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # weight slightly by brightness so the shuttle beats dim reflections
        resp = resp * (gray_roi.astype(np.float32) / 255.0)
        k = 9
        local_max = cv2.dilate(resp, np.ones((k, k), np.uint8))
        peaks = (resp >= local_max - 1e-6) & (resp > 0.45)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []
        # Shape gate: a *painted line* whose antialiased edge flickers looks
        # like motion, and is white, so it passes the colour test.  Its motion
        # component is long and thin, unlike a ball's compact blob.  Rejecting
        # by component aspect/area removes line flicker, the single most common
        # false positive near the court markings.
        comp_bin = (resp > 0).astype(np.uint8)
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(comp_bin, connectivity=8)
        ok_comp = np.zeros(n_lab, dtype=bool)
        if n_lab > 1:
            areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
            ws = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
            hs = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
            aspect = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1.0)
            ok_comp = (areas >= 6) & (areas <= 0.35 * gray_roi.size) & (aspect <= self.max_aspect)
        keep = [i for i in range(len(xs)) if ok_comp[lab[ys[i], xs[i]]]]
        if not keep:
            return []
        ys, xs = ys[keep], xs[keep]
        out: List[Detection] = []
        order = np.argsort(-resp[ys, xs])[: self.max_candidates]
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if not self._isolated(motion_roi, x, y):
                continue
            conf = float(np.clip(resp[y, x], 0, 1)) * 0.85
            if hint is not None:
                dist = float(np.hypot(ox + x - hint[0], oy + y - hint[1]))
                conf *= float(np.clip(1.25 - dist / (self.roi_px * 1.6), 0.05, 1.0))
            if conf < self.min_conf:
                continue
            out.append(Detection(xy=(float(ox + x), float(oy + y)), conf=conf,
                                 label="ball", box=(ox + x - 4.0, oy + y - 4.0,
                                                    ox + x + 4.0, oy + y + 4.0)))
        return out

    def reset(self) -> None:
        self._prev_gray = None
        self._prev2_gray = None
        self._frame_idx = 0
        self._hot = None
        # the background model is intentionally kept across resets: it describes
        # the camera, not the ball


class YoloBallDetector(BaseBallDetector):
    """Adapter around ``ultralytics.YOLO`` (or an exported ONNX model)."""

    name = "yolo"

    def __init__(self, weights: str, conf: float = 0.25, device: str = "cpu",
                 imgsz: int = 640, class_map: Optional[dict] = None):
        self.weights = weights
        self.conf = float(conf)
        self.device = device
        self.imgsz = int(imgsz)
        self.class_map = class_map or {0: "ball"}
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is not installed. `pip install ultralytics`, or run "
                "`python -m smartgym.cli export-onnx` and use the ONNX backend."
            ) from exc
        self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame: np.ndarray, hint: Optional[Point] = None) -> List[Detection]:
        model = self._load()
        results = model.predict(frame, conf=self.conf, device=self.device,
                                imgsz=self.imgsz, verbose=False)
        out: List[Detection] = []
        for res in results:
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0]) if hasattr(b, "cls") else 0
                label = self.class_map.get(cls_id, str(names.get(cls_id, cls_id)))
                if label != "ball":
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                c = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                out.append(Detection(xy=((x1 + x2) / 2, (y1 + y2) / 2), conf=c,
                                     label="ball", box=(x1, y1, x2, y2)))
        out.sort(key=lambda d: -d.conf)
        return out


def build_ball_detector(cfg: dict) -> Optional[BaseBallDetector]:
    det = (cfg or {}).get("detector", {}) if cfg else {}
    kind = str(det.get("ball", "classical")).lower()
    if kind in ("none", "off", "false"):
        return None
    if kind == "yolo":
        weights = det.get("weights")
        if not weights:
            raise ValueError("detector.weights is required for the yolo backend")
        return YoloBallDetector(weights, conf=float(det.get("conf", 0.25)),
                                device=str(det.get("device", "cpu")),
                                imgsz=int(det.get("imgsz", 640)))
    profile = det.get("profile")
    if profile is None:
        profile = "tennis_ball" if str((cfg or {}).get("sport", "")).lower() == "tennis" else "shuttle"
    return ClassicalBallDetector(profile=profile,
                                 roi_px=int(det.get("roi_px", 130)),
                                 min_conf=float(det.get("min_conf", 0.12)))
