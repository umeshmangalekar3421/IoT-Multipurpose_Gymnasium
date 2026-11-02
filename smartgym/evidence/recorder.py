"""Proof-image recorder.

For every point the system writes *evidence* to disk: the frame that decided the
call, the trajectory that led to it, a zoomed view of the landing point and the
scoreboard before/after.  That gives a human reviewer (coach, referee or
examiner) everything needed to accept or overturn an automated decision — which
is the difference between "an AI that prints a score" and "a system you can
audit".

Files per point (JPEG, quality configurable)::

    point_003_landing.jpg     decisive frame + zoom + call + score
    point_003_trail.jpg       full rally trajectory on one frame
    point_003_scoreboard.jpg  scoreboard after the point

Plus ``match_timeline.jpg`` (proof sheet) and ``score_timeline.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Event, EventType, PointRecord, ScoreState, TrackPoint
from ..court.model import CourtModel
from .annotate import (
    draw_ball,
    draw_court_overlay,
    draw_event_tags,
    draw_landing_marker,
    draw_text_block,
    draw_trail,
    draw_zoom_inset,
    render_scoreboard,
    render_summary,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class EvidenceConfig:
    out_dir: str = "runs/match"
    pre_frames: int = 15
    post_frames: int = 40
    jpeg_quality: int = 90
    annotate: bool = True
    save_scoreboard_png: bool = True
    save_trail: bool = True
    show_court_overlay: bool = True

    @classmethod
    def from_cfg(cls, cfg: Optional[dict]) -> "EvidenceConfig":
        ev = ((cfg or {}).get("evidence") or {}) if cfg else {}
        return cls(
            out_dir=str(ev.get("out_dir", "runs/match")),
            pre_frames=int(ev.get("pre_frames", 15)),
            post_frames=int(ev.get("post_frames", 40)),
            jpeg_quality=int(ev.get("jpeg_quality", 90)),
            annotate=bool(ev.get("annotate", True)),
            save_scoreboard_png=bool(ev.get("save_scoreboard_png", True)),
            save_trail=bool(ev.get("save_trail", True)),
            show_court_overlay=bool(ev.get("show_court_overlay", True)),
        )


class EvidenceRecorder:
    """Writes annotated proof images for each scored point."""

    def __init__(self, court: CourtModel, cfg: Optional[dict] = None,
                 out_dir: Optional[str] = None, mode: str = "singles"):
        self.court = court
        self.cfg = EvidenceConfig.from_cfg(cfg)
        if out_dir:
            self.cfg.out_dir = out_dir
        self.mode = mode
        self.dir = os.path.join(self.cfg.out_dir, "evidence")
        self.players: Sequence[str] = []
        self._track_by_frame: Dict[int, TrackPoint] = {}
        self._created: List[str] = []

    # -- setup --------------------------------------------------------------
    def prepare(self, players: Sequence[str], track: Sequence[TrackPoint]) -> None:
        os.makedirs(self.dir, exist_ok=True)
        self.players = list(players)
        self._track_by_frame = {tp.frame: tp for tp in track if not tp.interpolated}

    # -- public API ---------------------------------------------------------
    def record_points(self, video, points: Sequence[PointRecord],
                      fps: float = 30.0) -> Dict[int, List[str]]:
        """Second pass over the video, writing evidence for every point."""
        written: Dict[int, List[str]] = {}
        for point in points:
            try:
                files = self.record_point(video, point, fps=fps)
            except Exception as exc:  # evidence must never kill the analysis
                files = []
                print(f"[evidence] point {point.index} failed: {exc}")
            point.evidence = files
            written[point.index] = files
        return written

    def record_point(self, video, point: PointRecord, fps: float = 30.0) -> List[str]:
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv required")
        files: List[str] = []
        dec_frame = point.frame_end
        start = max(0, point.frame_start - self.cfg.pre_frames)
        end = dec_frame + self.cfg.post_frames

        trail = self._trail_points(point.frame_start, dec_frame)
        frames = list(video.read_window(start, end))
        if not frames:
            return files

        clean_dec = next((f for i, f in frames if i == dec_frame), frames[-1][1])
        clean_mid = frames[len(frames) // 2][1]

        # 1) decisive landing frame
        landing = self._find_landing(point)
        img = clean_dec.copy()
        if self.cfg.annotate:
            self._annotate(img, point, trail, landing, fps=fps, clean=clean_dec)
        files.append(self._write(img, f"point_{point.index:03d}_landing.jpg"))

        # 2) rally trajectory composite
        if self.cfg.save_trail:
            img2 = clean_mid.copy()
            if self.cfg.annotate:
                self._annotate(img2, point, trail, landing, fps=fps,
                               clean=clean_mid, trail_only=True)
            files.append(self._write(img2, f"point_{point.index:03d}_trail.jpg"))

        # 3) scoreboard snapshot
        if self.cfg.save_scoreboard_png and point.score_after is not None:
            sb = render_scoreboard(
                point.score_after, self.players or ["P1", "P2"],
                title=f"{point.score_after.sport.value} | point {point.index}")
            files.append(self._write(sb, f"point_{point.index:03d}_scoreboard.jpg"))
        return files

    # -- internals ----------------------------------------------------------
    def _trail_points(self, f0: int, f1: int) -> List[Tuple[float, float]]:
        pts = [self._track_by_frame[f].xy for f in range(f0, f1 + 1)
               if f in self._track_by_frame]
        return [(float(x), float(y)) for x, y in pts]

    @staticmethod
    def _find_landing(point: PointRecord) -> Optional[Event]:
        for ev in reversed(point.events):
            if ev.type in (EventType.BOUNCE_IN, EventType.BOUNCE_OUT,
                           EventType.NET_CONTACT, EventType.OUT_OF_PLAY):
                return ev
        for ev in reversed(point.events):
            if ev.type is EventType.LANDING:
                return ev
        return None

    def _annotate(self, frame: np.ndarray, point: PointRecord,
                  trail: Sequence[Tuple[float, float]], landing: Optional[Event],
                  fps: float, clean: np.ndarray, trail_only: bool = False) -> None:
        if self.cfg.show_court_overlay:
            draw_court_overlay(frame, self.court, self.mode)

        if not trail_only:
            draw_trail(frame, trail)
            for ev in point.events:
                if ev.type in (EventType.SHOT, EventType.SERVE) and ev.img_xy:
                    draw_ball(frame, ev.img_xy, radius=5,
                              color=(255, 200, 0) if ev.type is EventType.SERVE else (0, 200, 255))
            if landing is not None and landing.img_xy is not None:
                in_bounds = bool(landing.in_bounds) if landing.in_bounds is not None else (
                    landing.type is EventType.BOUNCE_IN)
                label = "IN" if in_bounds else "OUT"
                if landing.type is EventType.NET_CONTACT:
                    label = "NET"
                dist = ""
                if landing.meta and "distance_to_line_m" in landing.meta:
                    dist = f"{float(landing.meta['distance_to_line_m']) * 100:.0f} cm from line"
                elif landing.court_xy:
                    dist = f"({landing.court_xy[0]:.2f}, {landing.court_xy[1]:.2f}) m"
                draw_landing_marker(frame, landing.img_xy, in_bounds, label, dist)
                draw_zoom_inset(frame, landing.img_xy, clean)
        else:
            draw_trail(frame, trail, thickness=4, fade=False)
            if landing is not None and landing.img_xy is not None:
                draw_ball(frame, landing.img_xy, color=(0, 0, 255), radius=9)

        before = point.score_before.display if point.score_before else "?"
        after = point.score_after.display if point.score_after else "?"
        head = [
            f"POINT {point.index}   t={point.t_end:.2f}s   frame={point.frame_end}   "
            f"conf={point.confidence:.2f}" + ("   NEEDS REVIEW" if point.needs_review else ""),
            f"winner: {point.winner}   cause: {point.cause}",
            f"score: {before}   ->   {after}",
        ]
        draw_text_block(frame, head, origin=(14, 34), font_scale=0.62)
        if not trail_only:
            draw_event_tags(frame, point.events, origin=(14, 132))

    def _write(self, img: np.ndarray, name: str) -> str:
        path = os.path.join(self.dir, name)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
        cv2.imwrite(path, img, params)
        self._created.append(path)
        return os.path.relpath(path, self.cfg.out_dir)

    # -- match level --------------------------------------------------------
    def write_summary(self, result_dict: Dict[str, Any]) -> str:
        os.makedirs(self.dir, exist_ok=True)
        img = render_summary(result_dict)
        return self._write(img, "match_timeline.jpg")

    def write_timeline_json(self, result_dict: Dict[str, Any]) -> str:
        path = os.path.join(self.cfg.out_dir, "score_timeline.json")
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)
        return path

    def write_timeline_csv(self, points: Iterable[PointRecord]) -> str:
        path = os.path.join(self.cfg.out_dir, "score_timeline.csv")
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("index,t_start,t_end,frame_start,frame_end,winner,loser,cause,"
                     "outcome,confidence,needs_review,score_after,evidence\n")
            for p in points:
                fh.write(",".join([
                    str(p.index), f"{p.t_start:.3f}", f"{p.t_end:.3f}",
                    str(p.frame_start), str(p.frame_end), p.winner, p.loser,
                    p.cause, p.outcome.value, f"{p.confidence:.2f}",
                    "1" if p.needs_review else "0",
                    '"' + (p.score_after.display if p.score_after else "") + '"',
                    ";".join(p.evidence),
                ]) + "\n")
        return path
