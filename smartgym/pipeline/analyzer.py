"""End-to-end analyzer: video file in, official score + evidence out.

This is the orchestrator the CLI and the FastAPI service both call::

    analyzer = MatchAnalyzer.from_config("configs/badminton.yaml")
    result = analyzer.run("clip.mp4")
    print(result.final_score, result.winner, result.artifacts["summary"])

Stages
------
1. probe video (fps, size, duration)
2. calibrate the court (config corners → config homography → auto lines)
3. perceive (ball track in pixels and metres, plus player boxes)
4. extract physical events (bounces, hits, net contacts)
5. attribute strikes to players (optional, improves serve/point calls)
6. interpret rallies with the sport's rule engine (badminton / tennis)
7. write evidence images + JSON/CSV timelines
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.config import Config, load_config
from ..core.types import Event, EventType, PointRecord, ScoreState, Sport
from ..court.model import PRESETS, CourtModel, CourtSpec
from ..events.detector import EventDetector
from ..evidence.recorder import EvidenceRecorder
from ..perception.calibration import build_court
from ..perception.pipeline import PerceptionPipeline, PerceptionResult
from ..perception.players import refine_striker_sides
from ..perception.video import VideoSource
from ..sports.registry import get_rules
from ..analysis.rally import make_rally_engine

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class MatchResult:
    """Everything produced for one analysed clip."""

    meta: Dict[str, Any]
    calibration: Dict[str, Any]
    points: List[PointRecord] = field(default_factory=list)
    timeline: List[ScoreState] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def final_score(self) -> str:
        return self.timeline[-1].display if self.timeline else "0-0"

    @property
    def winner(self) -> Optional[str]:
        return self.timeline[-1].winner if self.timeline else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "smartgym.match.v1",
            "meta": self.meta,
            "calibration": self.calibration,
            "final_score": self.final_score,
            "winner": self.winner,
            "stats": self.stats,
            "warnings": self.warnings,
            "points": [p.as_dict() for p in self.points],
            "timeline": [s.as_dict() for s in self.timeline],
            "events": [e.as_dict() for e in self.events],
            "artifacts": self.artifacts,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)
        return path


class MatchAnalyzer:
    """Configure once, analyse many clips."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        sport = str(cfg.get("sport", "badminton")).lower()
        self.rules = get_rules(sport)
        self.sport = sport
        self.spec = self._resolve_spec(sport, cfg)
        self.mode = str(cfg.get_path("match.format", "singles"))
        self.warnings: List[str] = []

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, path: Optional[str] = None,
                    overrides: Optional[Dict[str, Any]] = None) -> "MatchAnalyzer":
        return cls(load_config(path, overrides))

    @staticmethod
    def _resolve_spec(sport: str, cfg: Config) -> CourtSpec:
        spec = PRESETS.get(sport, PRESETS["badminton"])
        overrides = cfg.get("court") or {}
        if not overrides:
            return spec
        # allow custom/multi-purpose courts with non-standard dimensions
        fields = {f: getattr(spec, f) for f in
                  ("length", "width", "singles_width", "service_line_from_net",
                   "doubles_long_service_inset")}
        fields.update({k: float(v) for k, v in overrides.items() if k in fields})
        return CourtSpec(sport=spec.sport, has_centre_line=spec.has_centre_line,
                         net_height_m=spec.net_height_m,
                         doubles_serve_uses_long_service_line=spec.doubles_serve_uses_long_service_line,
                         serve_uses_full_width_in_doubles=spec.serve_uses_full_width_in_doubles,
                         **fields)

    def _players(self) -> List[str]:
        players = self.cfg.get_path("match.players", ["P1", "P2"]) or ["P1", "P2"]
        if self.mode == "singles":
            players = players[:2]
        return list(players)

    # -- main ---------------------------------------------------------------
    def run(self, video_path: str, out_dir: Optional[str] = None,
            oracle_track: Optional[Sequence[Any]] = None) -> MatchResult:
        """Analyse one clip.

        ``oracle_track`` (a list of ``TrackPoint``) bypasses the ball detector and
        feeds a known trajectory straight into the event layer — the validation
        mode used by the benchmark and the demo to exercise scoring and evidence
        with detection held constant.
        """
        t0 = time.time()
        cfg = self.cfg
        out_dir = out_dir or str(cfg.get_path("evidence.out_dir", "runs/match"))
        os.makedirs(out_dir, exist_ok=True)

        video = VideoSource(
            video_path,
            default_fps=float(cfg.get_path("video.fps", None) or 30.0),
            max_frames=cfg.get_path("video.max_frames"),
            stride=int(cfg.get_path("video.stride", 1) or 1),
        )
        try:
            first = video.grab_first_frame()
            extra = [video.read_frame(i) for i in (10, 30, 60) if video.n_frames > i]
            extra = [f for f in extra if f is not None]
            court, cal_info = build_court(cfg.get("camera") or {}, self.spec,
                                          frame=first, extra_frames=extra)
            if not first is None and not cal_info.get("registered", False):
                self.warnings.append("court not calibrated: in/out calls are relative")
            if cal_info.get("method") == "assumed_full_frame":
                self.warnings.append(
                    "auto-calibration failed; assuming the court fills the frame. "
                    "Provide camera.court_corners_img for accurate line calls.")

            perception = PerceptionPipeline(cfg, court, video,
                                            track_override=list(oracle_track)
                                            if oracle_track is not None else None).run()

            detector = EventDetector(court, fps=video.fps, cfg=cfg,
                                     classifier=self._load_classifier())
            det = detector.detect(perception.track, n_frames=video.meta.n_frames)
            events = det.events

            if perception.players and bool(cfg.get_path("detector.players", "none")
                                           not in ("none", "off", False, "")):
                events = refine_striker_sides(events, perception.players)

            scorer = self.rules.new_match(
                self._players(), mode=self.mode,
                best_of=int(cfg.get_path("match.best_of", 3)),
                first_server=cfg.get_path("match.first_server"),
            )
            engine = make_rally_engine(self.sport, self.rules, scorer, court,
                                       mode=self.mode, cfg=cfg)
            points = engine.process(events)
            engine.finalize_rally()

            result = MatchResult(
                meta={
                    "sport": self.sport,
                    "format": self.mode,
                    "players": " vs ".join(self._players()),
                    "video": os.path.basename(video_path),
                    "fps": round(video.fps, 3),
                    "frames": video.meta.n_frames,
                    "duration_s": round(video.meta.duration_s, 2),
                    "analysed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_s": None,
                    "processing_fps": None,
                },
                calibration=cal_info,
                points=points,
                timeline=engine.timeline,
                events=events,
                stats={**perception.stats, **det.stats,
                       "n_points": len(points),
                       "n_rallies_scored": len(points)},
                warnings=list(self.warnings) + list(engine.warnings),
            )

            # ---- evidence ------------------------------------------------
            if bool(cfg.get_path("evidence.enabled", True)):
                recorder = EvidenceRecorder(court, cfg, out_dir=out_dir, mode=self.mode)
                recorder.prepare(self._players(), perception.track)
                recorder.record_points(video, points, fps=video.fps)
                result.artifacts["summary"] = recorder.write_summary(result.as_dict())
                result.artifacts["timeline_json"] = recorder.write_timeline_json(result.as_dict())
                result.artifacts["timeline_csv"] = recorder.write_timeline_csv(points)

            # ---- annotated video (optional, third pass) -------------------
            ann_path = cfg.get_path("output.annotated_video")
            if ann_path:
                result.artifacts["annotated_video"] = self._write_annotated_video(
                    video, court, perception, points, ann_path)

            result.meta["elapsed_s"] = round(time.time() - t0, 2)
            result.meta["processing_fps"] = round(
                video.meta.n_frames / max(1e-6, time.time() - t0), 1)

            match_json = cfg.get_path("output.timeline_json") or os.path.join(out_dir, "match.json")
            result.save(match_json)
            result.artifacts["match_json"] = match_json
            return result
        finally:
            video.release()

    # -- optional ML event classifier --------------------------------------
    def _load_classifier(self):
        if str(self.cfg.get_path("events.model", "rules")).lower() not in ("mlp", "ml", "model"):
            return None
        path = self.cfg.get_path("events.model_path")
        if not path or not os.path.exists(path):
            self.warnings.append("events.model_path missing: falling back to pure rules")
            return None
        try:
            from ..training.event_model import EventClassifier

            return EventClassifier.load(path)
        except Exception as exc:
            self.warnings.append(f"could not load event model ({exc}); using rules")
            return None

    # -- annotated video renderer ------------------------------------------
    def _write_annotated_video(self, video: VideoSource, court: CourtModel,
                               perception: PerceptionResult,
                               points: Sequence[PointRecord], path: str) -> str:
        if cv2 is None:  # pragma: no cover
            return ""
        from ..evidence.annotate import (
            draw_ball, draw_court_overlay, draw_landing_marker, draw_text_block,
            draw_trail,
        )

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, video.fps,
                                 (video.meta.width, video.meta.height))
        track = {tp.frame: tp for tp in perception.track}
        landings = {}
        for p in points:
            for ev in p.events:
                if ev.type in (EventType.BOUNCE_IN, EventType.BOUNCE_OUT) and ev.img_xy:
                    landings[ev.frame] = ev
        current_score = None
        for idx, t, frame in video.frames():
            draw_court_overlay(frame, court, self.mode)
            recent = [track[f].xy for f in range(max(0, idx - 20), idx + 1) if f in track]
            draw_trail(frame, recent)
            if idx in track:
                draw_ball(frame, track[idx].xy)
            if idx in landings:
                ev = landings[idx]
                draw_landing_marker(frame, ev.img_xy,
                                    bool(ev.in_bounds) if ev.in_bounds is not None else True,
                                    "IN" if ev.in_bounds else "OUT")
            for p in points:
                if p.frame_end <= idx:
                    current_score = p.score_after
            if current_score is not None:
                draw_text_block(frame, [f"{current_score.display}", f"t={t:6.2f}s"],
                                origin=(14, 30), font_scale=0.7)
            writer.write(frame)
        writer.release()
        return path
