"""Perception pipeline: video -> ball track + player boxes in court metres."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ..core.kalman import BallTracker, interpolate_gaps
from .linker import Candidate, OfflineLinker, to_track_points
from ..core.types import Detection, PlayerBox, Point, TrackPoint
from ..court.model import CourtModel
from .ball import BaseBallDetector, build_ball_detector
from .players import PlayerTracker, build_player_tracker
from .video import VideoMeta, VideoSource


@dataclass
class PerceptionResult:
    meta: VideoMeta
    track: List[TrackPoint]
    players: Dict[int, List[PlayerBox]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def frames_with_ball(self) -> int:
        return len([tp for tp in self.track if not tp.interpolated])


class PerceptionPipeline:
    """Runs detection + tracking frame by frame (streaming friendly)."""

    def __init__(self, cfg: dict, court: CourtModel, video: VideoSource,
                 detector: Optional[BaseBallDetector] = None,
                 player_tracker: Optional[PlayerTracker] = None,
                 track_override: Optional[List[TrackPoint]] = None):
        self.track_override = track_override
        self.cfg = cfg or {}
        self.court = court
        self.video = video
        self.detector = detector if detector is not None else build_ball_detector(self.cfg)
        self.player_tracker = (player_tracker if player_tracker is not None
                               else build_player_tracker(self.cfg, court))
        tracking = (self.cfg.get("tracking") or {})
        self.tracker = BallTracker(
            dt=1.0 / max(1e-3, video.fps),
            gate_px=float(tracking.get("gate_px", 90.0)),
            max_coast_frames=int(tracking.get("max_coast_frames", 12)),
            reacquire_px=float(tracking.get("reacquire_px", 220.0)),
        )
        self.prefilter = bool((self.cfg.get("detector") or {}).get("prefilter", True))
        tracking_cfg = self.cfg.get("tracking") or {}
        self.offline = str(tracking_cfg.get("mode", "offline")).lower() != "online"
        self.linker = OfflineLinker(
            fps=video.fps,
            max_gap=int(tracking_cfg.get("max_coast_frames", 12)),
            max_speed_px_s=float(tracking_cfg.get("max_speed_px_s", 3200.0)),
            gate_px=float(tracking_cfg.get("gate_px", 90.0)) * 1.6,
            min_track_points=int(tracking_cfg.get("min_track_points", 6)),
            min_displacement_px=float(tracking_cfg.get("min_displacement_px", 90.0)),
            min_mean_speed_px_s=float(tracking_cfg.get("min_mean_speed_px_s", 110.0)),
        )

    def run(self) -> PerceptionResult:
        """Perceive the whole clip.

        In ``offline`` mode (the default for files) every frame is scanned and
        the candidates are linked globally afterwards — far more robust than
        online association.  In ``online`` mode the Kalman tracker runs frame by
        frame, which is what a live RTSP/CSI stream needs.
        """
        if self.track_override is not None:
            return self._run_override()
        if self.offline:
            return self._run_offline()
        return self._run_online()

    # -- oracle / supplied track (validation mode) --------------------------
    def _run_override(self) -> PerceptionResult:
        track = list(self.track_override)
        for tp in track:
            if tp.court_xy is None and self.court.registered:
                tp.court_xy = self.court.to_court_point(tp.xy)
        stats = {
            "frames_processed": self.video.n_frames,
            "frames_with_detection": len(track),
            "frames_with_track": len(track),
            "frames_with_real_track": len(track),
            "track_coverage": round(len(track) / max(1, self.video.n_frames), 4),
            "detection_rate": round(len(track) / max(1, self.video.n_frames), 4),
            "court_points": sum(1 for tp in track if tp.court_xy is not None),
            "players_detected": 0,
            "fps": self.video.fps,
            "tracking_mode": "oracle",
        }
        return PerceptionResult(meta=self.video.meta, track=track, players={}, stats=stats)

    # -- offline (files) ----------------------------------------------------
    def _run_offline(self) -> PerceptionResult:
        candidates: Dict[int, List[Candidate]] = {}
        players: Dict[int, List[PlayerBox]] = {}
        n_frames = 0
        n_detected = 0

        if self.detector is not None:
            self._initialize_detector()

        for idx, t, frame in self.video.frames():
            n_frames += 1
            dets: List[Detection] = []
            if self.detector is not None:
                try:
                    dets = self.detector.detect(frame, hint=None)
                except Exception as exc:  # pragma: no cover
                    self.cfg.setdefault("_warnings", []).append(
                        f"detector error at frame {idx}: {exc}")
                if dets:
                    n_detected += 1
                    candidates[idx] = [Candidate(frame=idx, t=t, xy=d.xy,
                                                 conf=d.conf, box=d.box)
                                       for d in dets]
                    candidates[idx].sort(key=lambda c: -c.conf)

            if self.player_tracker is not None and idx % self.player_tracker.sample_every == 0:
                boxes = self.player_tracker.step(frame, idx, t)
                if boxes:
                    players[idx] = boxes

        tracks = self.linker.run(candidates)
        track = to_track_points(tracks, self.court)
        track = interpolate_gaps(track,
                                 max_gap=int((self.cfg.get("tracking") or {}).get(
                                     "max_coast_frames", 12)))
        for tp in track:
            if tp.court_xy is None and self.court.registered:
                cxy = self.court.to_court_point(tp.xy)
                if cxy is not None:
                    tp.court_xy = cxy

        real = len([tp for tp in track if not tp.interpolated])
        stats = {
            "frames_processed": n_frames,
            "frames_with_detection": n_detected,
            "frames_with_track": len(track),
            "frames_with_real_track": real,
            "track_coverage": round(real / max(1, n_frames), 4),
            "detection_rate": round(n_detected / max(1, n_frames), 4),
            "court_points": sum(1 for tp in track if tp.court_xy is not None),
            "players_detected": len(players),
            "fps": self.video.fps,
            "tracking_mode": "offline",
            "candidate_tracks": len(tracks),
            "tracklets": [{"first": tr.first_frame, "last": tr.last_frame,
                           "points": tr.length, "score": round(tr.score, 2)}
                          for tr in tracks],
        }
        return PerceptionResult(meta=self.video.meta, track=track,
                                players=players, stats=stats)

    def _initialize_detector(self, n_samples: int = 24) -> None:
        """Sample frames across the clip to give the detector a background model."""
        try:
            total = self.video.n_frames or 0
            if total < 10:
                return
            step = max(1, total // n_samples)
            frames = [f for i, f in self.video.read_window(0, total)
                      if i % step == 0][:n_samples]
            if frames:
                self.detector.initialize(frames)
        except Exception:
            pass
        finally:
            # rewind for the main pass
            try:
                self.video._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass

    # -- online (live streams) ---------------------------------------------
    def _run_online(self) -> PerceptionResult:
        track: List[TrackPoint] = []
        players: Dict[int, List[PlayerBox]] = {}
        n_detected = 0
        n_frames = 0

        for idx, t, frame in self.video.frames():
            n_frames += 1
            dets: List[Detection] = []
            if self.detector is not None:
                hint: Optional[Point] = None
                if self.tracker._kf is not None and self.tracker._lost <= self.tracker.max_coast_frames:
                    hint = self.tracker._kf.pos
                try:
                    dets = self.detector.detect(frame, hint=hint)
                except Exception as exc:  # pragma: no cover
                    dets = []
                    self.cfg.setdefault("_warnings", []).append(f"detector error at {idx}: {exc}")
                if dets:
                    n_detected += 1

            tp = self.tracker.step(idx, t, dets)
            if tp is not None:
                if self.court.registered:
                    cxy = self.court.to_court_point(tp.xy)
                    if cxy is not None:
                        if abs(cxy[0]) < 5 * self.court.spec.length and \
                                abs(cxy[1]) < 5 * self.court.spec.width:
                            tp.court_xy = cxy
                track.append(tp)

            if self.player_tracker is not None and idx % self.player_tracker.sample_every == 0:
                boxes = self.player_tracker.step(frame, idx, t)
                if boxes:
                    players[idx] = boxes

        track = interpolate_gaps(track,
                                 max_gap=int((self.cfg.get("tracking") or {}).get("max_coast_frames", 12)))
        for tp in track:
            if tp.court_xy is None and self.court.registered:
                cxy = self.court.to_court_point(tp.xy)
                if cxy is not None:
                    tp.court_xy = cxy

        real = len([tp for tp in track if not tp.interpolated])
        stats = {
            "frames_processed": n_frames,
            "frames_with_detection": n_detected,
            "frames_with_track": len(track),
            "frames_with_real_track": real,
            "track_coverage": round(real / max(1, n_frames), 4),
            "detection_rate": round(n_detected / max(1, n_frames), 4),
            "court_points": sum(1 for tp in track if tp.court_xy is not None),
            "players_detected": len(players),
            "fps": self.video.fps,
            "tracking_mode": "online",
        }
        return PerceptionResult(meta=self.video.meta, track=track,
                                players=players, stats=stats)
