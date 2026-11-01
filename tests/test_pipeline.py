"""Perception, event extraction, rally interpretation and evidence."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from smartgym.analysis.rally import make_rally_engine
from smartgym.core.kalman import BallTracker, interpolate_gaps, split_tracklets
from smartgym.core.types import Detection, Event, EventType, Side, TrackPoint
from smartgym.court.model import BADMINTON, TENNIS, CourtModel
from smartgym.events.detector import EventDetector
from smartgym.events.trajectory import (
    build_segments,
    detect_bounces,
    segment_contacts,
)
from smartgym.perception.linker import Candidate, OfflineLinker
from smartgym.sports.registry import get_rules


def metres_court(spec=BADMINTON):
    return CourtModel(spec, np.eye(3))


def parabola_track(frame0: int, n: int, fps: float, x0: float, x1: float,
                   apex_y: float, ground_y: float, seed: int = 0,
                   jitter: float = 0.0) -> list:
    """A synthetic arc in PIXELS: rises to ``apex_y`` then falls to ``ground_y``."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        f = k / max(1, n - 1)
        y = apex_y + (ground_y - apex_y) * f ** 2
        x = x0 + (x1 - x0) * f
        out.append(TrackPoint(frame=frame0 + k, t=(frame0 + k) / fps,
                              xy=(x + rng.normal(0, jitter), y + rng.normal(0, jitter)),
                              conf=1.0))
    return out


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
class TestTracker:
    def test_tracks_a_moving_ball(self):
        tracker = BallTracker(dt=1 / 30, gate_px=60)
        seen = []
        for k in range(20):
            tp = tracker.step(k, k / 30, [Detection(xy=(100 + 12 * k, 200.0), conf=0.9)])
            seen.append(tp)
        assert all(tp is not None for tp in seen)
        assert seen[-1].xy[0] > seen[0].xy[0]

    def test_rejects_static_distractor(self):
        """A detector locked onto a stationary object must drop the track."""
        tracker = BallTracker(dt=1 / 30, gate_px=60, max_static_frames=5,
                              stasis_window=4, stasis_px=20)
        results = [tracker.step(k, k / 30, [Detection(xy=(500.0, 300.0), conf=0.9)])
                   for k in range(30)]
        assert any(r is None for r in results), "static track was never dropped"

    def test_coasts_through_occlusion(self):
        tracker = BallTracker(dt=1 / 30, gate_px=80, max_coast_frames=5)
        for k in range(5):
            tracker.step(k, k / 30, [Detection(xy=(100 + 10 * k, 200.0))])
        tp = tracker.step(5, 5 / 30, [])          # occluded
        assert tp is not None and tp.interpolated

    def test_interpolate_gaps(self):
        track = [TrackPoint(frame=0, t=0.0, xy=(0, 0)),
                 TrackPoint(frame=4, t=4 / 30, xy=(40, 0))]
        filled = interpolate_gaps(track, max_gap=6)
        assert [p.frame for p in filled] == [0, 1, 2, 3, 4]
        assert filled[2].xy[0] == pytest.approx(20.0)

    def test_split_tracklets_on_gap(self):
        a = [TrackPoint(frame=i, t=i / 30, xy=(i, 0)) for i in range(5)]
        b = [TrackPoint(frame=20 + i, t=(20 + i) / 30, xy=(20 + i, 0)) for i in range(5)]
        assert len(split_tracklets(a + b, max_gap=3)) == 2


class TestOfflineLinker:
    def test_links_a_clean_arc(self):
        cands = {i: [Candidate(i, i / 30.0, (100 + 8 * i, 300.0 - 2 * i), 1.0)]
                 for i in range(30)}
        linker = OfflineLinker(fps=30.0, min_track_points=5,
                               min_displacement_px=20, min_mean_speed_px_s=10)
        tracks = linker.run(cands)
        assert len(tracks) == 1
        assert tracks[0].length == 30

    def test_discards_parked_candidates(self):
        cands = {i: [Candidate(i, i / 30.0, (500.0, 300.0), 1.0)] for i in range(40)}
        linker = OfflineLinker(fps=30.0, min_mean_speed_px_s=100)
        assert linker.run(cands) == []

    def test_prefers_the_consistent_candidate(self):
        rng = np.random.default_rng(0)
        cands = {}
        for i in range(30):
            ball = Candidate(i, i / 30.0, (100 + 10 * i, 200.0), 0.9)
            noise = Candidate(i, i / 30.0,
                              (rng.uniform(0, 1200), rng.uniform(0, 700)), 0.9)
            cands[i] = [ball, noise]
        linker = OfflineLinker(fps=30.0, min_track_points=5,
                               min_displacement_px=50, min_mean_speed_px_s=50)
        tracks = linker.run(cands)
        assert tracks and tracks[0].length > 25
        assert tracks[0].points[0].xy[1] == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# Event extraction
# --------------------------------------------------------------------------- #
class TestEventDetector:
    def test_detects_a_bounce(self):
        """A descending arc that ends on the floor is one floor contact."""
        track = parabola_track(0, 30, 30.0, x0=300, x1=700, apex_y=100, ground_y=600)
        court = metres_court()
        for tp in track:
            tp.court_xy = (6.7, 3.0)
        det = EventDetector(court, fps=30.0, cfg={})
        result = det.detect(track, n_frames=30)
        assert result.stats["n_segments"] >= 1
        assert any(c.kind in ("bounce", "lost") for c in result.contacts)

    def test_segments_split_on_time_gap(self):
        a = parabola_track(0, 20, 30.0, 100, 400, 100, 500)
        b = parabola_track(40, 20, 30.0, 800, 500, 120, 520)
        for tp in a + b:
            tp.court_xy = (5.0, 3.0)
        segs = build_segments(a + b, max_gap=5, fps=30.0)
        assert len(segs) == 2

    def test_bounce_is_located_near_the_ground(self):
        """A descending arc that rebounds is exactly one floor contact."""
        falling = parabola_track(0, 30, 30.0, 400, 420, apex_y=120, ground_y=640)
        # rebound: the ball rises again after touching the floor
        rising = [TrackPoint(frame=30 + i, t=(30 + i) / 30.0,
                             xy=(420.0 + 4 * i, 640.0 - 18 * i), conf=1.0)
                  for i in range(12)]
        track = falling + rising
        for tp in track:
            tp.court_xy = (7.0, 3.0)
        segs = build_segments(track, max_gap=5, fps=30.0)
        bounces = detect_bounces(segs[0], min_prominence_px=5.0, min_spacing_frames=3)
        assert bounces, "no bounce found in a clean rebound"
        assert abs(bounces[0].img_xy[1] - 640) < 30

    def test_events_are_chronological(self):
        track = parabola_track(0, 30, 30.0, 300, 700, 100, 600)
        for tp in track:
            tp.court_xy = (8.0, 3.0)
        det = EventDetector(metres_court(), fps=30.0, cfg={})
        events = det.detect(track, n_frames=30).events
        assert [e.frame for e in events] == sorted(e.frame for e in events)

    def test_empty_track_is_safe(self):
        det = EventDetector(metres_court(), fps=30.0, cfg={})
        result = det.detect([], n_frames=100)
        assert result.events == [] and result.contacts == []


# --------------------------------------------------------------------------- #
# Rally interpretation
# --------------------------------------------------------------------------- #
def run_engine(sport, events, players=("P1", "P2"), mode="singles", spec=BADMINTON):
    rules = get_rules(sport)
    court = metres_court(spec)
    scorer = rules.new_match(list(players), mode=mode, best_of=3)
    engine = make_rally_engine(sport, rules, scorer, court, mode, {})
    points = engine.process(events)
    engine.finalize_rally()
    return points, scorer, engine


class TestRallyEngineBadminton:
    def test_landing_in_awards_the_striker(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.SHOT, 40, 1.33, side=Side.B, court_xy=(7.0, 2.0)),
            Event(EventType.LANDING, 70, 2.33, court_xy=(7.5, 1.0)),
        ]
        points, scorer, _ = run_engine("badminton", events)
        assert len(points) == 1
        assert points[0].winner == "P1"      # the ball landed in side B
        assert scorer.points["P1"] == 1

    def test_landing_out_awards_the_opponent(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.SHOT, 40, 1.33, side=Side.B, court_xy=(7.0, 2.0)),
            Event(EventType.SHOT, 70, 2.33, side=Side.A, court_xy=(5.0, 2.2)),
            Event(EventType.LANDING, 100, 3.33, court_xy=(13.9, 2.0)),
        ]
        points, scorer, _ = run_engine("badminton", events)
        assert points[0].winner == "P2"

    def test_net_fault(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.NET_CONTACT, 40, 1.33, court_xy=(6.8, 2.0)),
        ]
        points, _, _ = run_engine("badminton", events)
        assert points[0].winner == "P2" and points[0].cause == "net_fault"

    def test_serve_fault(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 40, 1.33, court_xy=(9.6, 2.0)),   # past the service line
        ]
        points, _, _ = run_engine("badminton", events)
        assert points[0].winner == "P2" and "serve" in points[0].cause

    def test_non_terminal_bounce_does_not_end_the_rally(self):
        bounce = Event(EventType.LANDING, 40, 1.33, court_xy=(7.5, 2.0))
        bounce.meta["terminal"] = False
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            bounce,
            Event(EventType.LANDING, 90, 3.0, court_xy=(13.9, 2.0)),
        ]
        points, _, _ = run_engine("badminton", events)
        assert len(points) == 1, "a survived bounce must not pay out a point"
        assert points[0].cause == "landing_out"

    def test_point_record_carries_provenance(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 70, 2.33, court_xy=(7.5, 1.0)),
        ]
        points, _, _ = run_engine("badminton", events)
        rec = points[0]
        assert rec.score_before is not None and rec.score_after is not None
        assert rec.score_before.display == "0-0"
        assert rec.score_after.display == "1-0"
        assert any(e.type is EventType.POINT for e in rec.events)
        assert rec.frame_start <= rec.frame_end
        json.dumps(rec.as_dict())


class TestRallyEngineTennis:
    def test_serve_fault_then_point(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 25, 0.83, court_xy=(9.0, 2.0)),     # short: fault
            Event(EventType.SHOT, 45, 1.5, side=Side.A, court_xy=(5.0, 2.0)),  # 2nd serve
            Event(EventType.LANDING, 60, 2.0, court_xy=(16.0, 2.0)),     # good serve
            Event(EventType.SHOT, 85, 2.83, side=Side.B, court_xy=(17.0, 2.0)),
            Event(EventType.LANDING, 100, 3.33, court_xy=(6.0, 2.0)),    # in, A's side
            Event(EventType.SHOT, 120, 4.0, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 145, 4.83, court_xy=(24.5, 2.0)),   # out beyond B
        ]
        points, scorer, _ = run_engine("tennis", events, spec=TENNIS)
        assert len(points) == 1
        assert points[0].cause == "landing_out"
        assert points[0].winner == "P2", "A hit the ball out, so B wins the point"

    def test_double_fault(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 25, 0.83, court_xy=(9.0, 2.0)),
            Event(EventType.SHOT, 45, 1.5, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 60, 2.0, court_xy=(25.0, 2.0)),
        ]
        points, _, _ = run_engine("tennis", events, spec=TENNIS)
        assert points[0].cause == "double_fault" and points[0].winner == "P2"

    def test_double_bounce(self):
        events = [
            Event(EventType.SERVE, 10, 0.33, side=Side.A, court_xy=(5.0, 2.0)),
            Event(EventType.LANDING, 60, 2.0, court_xy=(16.0, 2.0)),
            Event(EventType.LANDING, 95, 3.17, court_xy=(17.0, 2.5)),
        ]
        points, _, _ = run_engine("tennis", events, spec=TENNIS)
        assert points[0].cause == "double_bounce"
        assert points[0].winner == "P1"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class TestEvidence:
    def test_evidence_images_are_written(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from smartgym.evidence.recorder import EvidenceRecorder

        video_path = str(tmp_path / "clip.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
        for i in range(60):
            frame = np.full((180, 320, 3), 40, np.uint8)
            cv2.circle(frame, (60 + 3 * i, 90), 5, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        from smartgym.core.types import PointRecord
        from smartgym.perception.video import VideoSource

        court = metres_court()
        recorder = EvidenceRecorder(court, {"evidence": {"out_dir": str(tmp_path)}},
                                    out_dir=str(tmp_path))
        track = [TrackPoint(frame=i, t=i / 30, xy=(60 + 3 * i, 90)) for i in range(60)]
        recorder.prepare(["P1", "P2"], track)
        record = PointRecord(index=1, t_start=0.2, t_end=1.5, frame_start=6, frame_end=45,
                             winner="P1", loser="P2", outcome=None, cause="landing_in",
                             events=[Event(EventType.BOUNCE_IN, 45, 1.5,
                                           court_xy=(7.0, 2.0), img_xy=(195.0, 90.0),
                                           in_bounds=True)])
        with VideoSource(video_path) as video:
            files = recorder.record_point(video, record)
        assert files, "no evidence written"
        for rel in files:
            assert os.path.exists(os.path.join(str(tmp_path), rel))

    def test_summary_and_timeline(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from smartgym.evidence.recorder import EvidenceRecorder

        recorder = EvidenceRecorder(metres_court(), {"evidence": {"out_dir": str(tmp_path)}},
                                    out_dir=str(tmp_path))
        payload = {"meta": {"sport": "badminton", "players": "P1 vs P2",
                            "video": "clip.mp4"},
                   "final_score": "1-0", "winner": "P1",
                   "points": [{"index": 1, "t_end": 1.5, "winner": "P1",
                               "cause": "landing_in",
                               "score_after": {"display": "1-0"},
                               "needs_review": False}]}
        assert os.path.exists(os.path.join(str(tmp_path), recorder.write_summary(payload)))
        jp = recorder.write_timeline_json(payload)
        with open(jp, encoding="utf-8") as fh:
            assert json.load(fh)["final_score"] == "1-0"
        assert os.path.exists(recorder.write_timeline_csv([]))


# --------------------------------------------------------------------------- #
# Config & registry
# --------------------------------------------------------------------------- #
class TestConfigAndRegistry:
    def test_config_defaults_and_overrides(self, tmp_path):
        from smartgym.core.config import load_config

        path = tmp_path / "cfg.yaml"
        path.write_text("sport: tennis\nmatch:\n  players: [Ann, Bob]\n")
        cfg = load_config(str(path))
        assert cfg["sport"] == "tennis"
        assert cfg.get_path("match.players") == ["Ann", "Bob"]
        assert cfg.get_path("events.line_margin_m") == 0.03      # default preserved
        cfg.set_path("a.b.c", 5)
        assert cfg.get_path("a.b.c") == 5

    def test_unknown_sport_is_rejected(self):
        with pytest.raises(KeyError):
            get_rules("cricket")

    def test_alias_resolution(self):
        assert get_rules("shuttle").sport.value == "badminton"
