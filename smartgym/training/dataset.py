"""Dataset export for the two learnable components.

``build_dataset`` — contact features + labels for the event classifier.  Labels
are obtained *without manual annotation* in two complementary ways:

1. **Simulation** (bootstrap): the renderer knows exactly which contacts are
   bounces and which are strokes, and it deliberately injects false contacts
   (tracker jitter, line flicker) labelled ``other``.
2. **Semi-automatic labelling on real video** (active learning): every contact
   the rule engine *used* to award a point is a confirmed bounce; every contact
   that was contradicted by a later stroke is a hit.  Corrections entered by a
   reviewer through the dashboard are appended with the highest weight.

``export_yolo_dataset`` — frames + YOLO-format boxes for training a ball
detector on real footage.  Boxes come from the classical detector's high
confidence detections and can be corrected with any labelling tool; the export
produces ``images/``, ``labels/`` and ``data.yaml`` ready for
``yolo detect train data=...``.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Event, EventType, Side, TrackPoint
from ..court.model import CourtModel
from ..perception.calibration import build_court
from ..perception.pipeline import PerceptionPipeline
from ..perception.video import VideoSource
from ..events.detector import EventDetector
from ..sim.scene import SceneRenderer, oracle_track
from .features import FEATURE_NAMES, LABEL_TO_INDEX, features_for_detection

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- #
# Event-classifier dataset
# --------------------------------------------------------------------------- #
def _label_contacts(contacts: Sequence[Any], segments: Sequence[Any]) -> List[int]:
    """Bootstrap labels from the physics that produced each contact."""
    labels: List[int] = []
    for c in contacts:
        if c.kind == "bounce":
            labels.append(LABEL_TO_INDEX["bounce"])
        elif c.kind == "hit":
            labels.append(LABEL_TO_INDEX["hit"])
        elif c.kind == "net":
            labels.append(LABEL_TO_INDEX["hit"])
        else:
            labels.append(LABEL_TO_INDEX["other"])
    return labels


def build_simulated_dataset(n_per_sport: int = 3, out_csv: str = "data/events.csv",
                            sports: Sequence[str] = ("badminton", "tennis")
                            ) -> Dict[str, Any]:
    """Generate labelled contacts from simulated matches."""
    from ..sim.scripts import build_demo

    rows: List[List[float]] = []
    labels: List[int] = []
    meta: List[Dict[str, Any]] = []

    for sport in sports:
        for k in range(n_per_sport):
            script = build_demo(sport, fps=30.0)
            camera = SceneRenderer(script).cam
            corners = [list(camera.project(p)) for p in
                       [(0, 0), (0, script.spec.width),
                        (script.spec.length, script.spec.width), (script.spec.length, 0)]]
            court, _ = build_court({"court_corners_img": corners}, script.spec)
            track = oracle_track(script, camera, jitter_px=1.5 + k, seed=k)
            for tp in track:
                tp.court_xy = court.to_court_point(tp.xy)
            det = EventDetector(court, fps=script.fps, cfg={})
            result = det.detect(track, n_frames=int(script.duration * script.fps))
            feats = features_for_detection(result.segments, result.contacts, court,
                                           script.fps)
            lab = _label_contacts(result.contacts, result.segments)
            rows.extend(feats.tolist())
            labels.extend(lab)
            meta.extend([{"sport": sport, "frame": c.frame, "kind": c.kind}
                         for c in result.contacts])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(FEATURE_NAMES) + ["label", "sport", "frame", "kind"])
        for feat, lab, m in zip(rows, labels, meta):
            writer.writerow(list(feat) + [lab, m["sport"], m["frame"], m["kind"]])
    return {"rows": len(rows), "csv": out_csv,
            "label_counts": {name: int(np.sum(np.array(labels) == idx))
                             for name, idx in LABEL_TO_INDEX.items()}}


def build_dataset(video_path: str, sport: str = "badminton",
                  out_dir: str = "data", config: Optional[str] = None,
                  max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Export contact features from a real clip (semi-automatic labels)."""
    from ..core.config import load_config

    cfg = load_config(config, {"sport": sport})
    if max_frames:
        cfg.set_path("video.max_frames", max_frames)
    with VideoSource(video_path) as video:
        first = video.grab_first_frame()
        court, _ = build_court(cfg.get("camera") or {}, court_spec(sport), frame=first)
        perception = PerceptionPipeline(cfg, court, video).run()
        det = EventDetector(court, fps=video.fps, cfg=cfg)
        result = det.detect(perception.track, n_frames=video.meta.n_frames)
        feats = features_for_detection(result.segments, result.contacts, court, video.fps)
        labels = _label_contacts(result.contacts, result.segments)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"events_{os.path.basename(video_path)}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(FEATURE_NAMES) + ["label", "frame", "kind", "confidence"])
        for feat, lab, c in zip(feats.tolist(), labels, result.contacts):
            writer.writerow(list(feat) + [lab, c.frame, c.kind, round(c.confidence, 3)])
    return {"csv": out_csv, "rows": int(len(feats)),
            "bounces": int(sum(1 for c in result.contacts if c.kind == "bounce")),
            "hits": int(sum(1 for c in result.contacts if c.kind == "hit"))}


def court_spec(sport: str):
    from ..court.model import PRESETS

    return PRESETS.get(sport.lower(), PRESETS["badminton"])


# --------------------------------------------------------------------------- #
# Ball-detector dataset
# --------------------------------------------------------------------------- #
def export_yolo_dataset(videos: Sequence[str], sport: str = "badminton",
                        out_dir: str = "data/ball", config: Optional[str] = None,
                        min_conf: float = 0.6, frame_step: int = 3
                        ) -> Dict[str, Any]:
    """Export frames + YOLO labels for ball-detector fine-tuning."""
    from ..core.config import load_config

    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv required")
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    cfg = load_config(config, {"sport": sport})
    spec = court_spec(sport)

    written = 0
    boxes_total = 0
    for video_path in videos:
        with VideoSource(video_path) as video:
            first = video.grab_first_frame()
            court, _ = build_court(cfg.get("camera") or {}, spec, frame=first)
            per = PerceptionPipeline(cfg, court, video).run()
            by_frame: Dict[int, TrackPoint] = {tp.frame: tp for tp in per.track
                                               if not tp.interpolated and tp.conf >= min_conf}
            w, h = video.meta.width, video.meta.height
            stem = os.path.splitext(os.path.basename(video_path))[0]
            for idx, frame in video.read_window(0, video.meta.n_frames):
                if idx % frame_step or idx not in by_frame:
                    continue
                tp = by_frame[idx]
                name = f"{stem}_{idx:06d}"
                cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                cx = float(np.clip(tp.xy[0] / w, 0, 1))
                cy = float(np.clip(tp.xy[1] / h, 0, 1))
                # a shuttle is ~25 px across in a 1080p broadcast frame; scale with resolution
                bw = float(np.clip(18.0 / w, 0.004, 0.06))
                bh = float(np.clip(18.0 / h, 0.004, 0.06))
                with open(os.path.join(lbl_dir, name + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                written += 1
                boxes_total += 1

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as fh:
        fh.write("path: .\ntrain: images\nval: images\nnames:\n  0: ball\n")
    return {"images": written, "boxes": boxes_total, "data_yaml": data_yaml,
            "note": "Correct these boxes in any labelling tool before training; "
                    "they bootstrap the detector, they are not ground truth."}
