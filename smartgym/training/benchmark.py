"""Accuracy benchmark against simulated ground truth.

A single "end-to-end accuracy" number hides where a system breaks.  This harness
reports the pipeline layer by layer, so each component can be defended
separately:

============  ==================================================================
``rules``     Official scoring run on *perfect* events.  Pure rule-engine
              accuracy — should be 100% and any failure is a rules bug.
``events``    Event extraction run on a *perfect ball track* (optionally with
              injected jitter / dropouts).  Measures the physics heuristics.
``end_to_end`` Rendered video through the full perception stack.  Measures the
              detector + tracker + events + rules together.
============  ==================================================================

Every layer also reports the number of rallies on which the *point winner* was
correct, which is the metric that actually matters for deciding a match.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..analysis.rally import make_rally_engine
from ..core.types import PointRecord
from ..court.model import BADMINTON, PRESETS, TENNIS, CourtModel
from ..events.detector import EventDetector
from ..sim.scene import SceneRenderer, perfect_events, oracle_track, write_ground_truth
from ..sim.scripts import build_badminton_demo, build_tennis_demo
from ..sports.registry import get_rules


def _compare(points: Sequence[PointRecord], rallies: Sequence[Any]) -> Dict[str, Any]:
    """Compare engine output with the script's reference results.

    Points are matched to rallies **by time window**, not by index: one missed
    rally would otherwise shift every later comparison and report a cascade of
    false failures.  A rally counts as recalled when at least one detected point
    ends inside its window; the winner is then compared on the *last* point in
    that window (which is the one that decided the rally).
    """
    got = sorted(points, key=lambda p: p.t_end)
    n = len(rallies)

    winner_ok = 0
    cause_ok = 0
    score_ok = 0
    recalled = 0
    used = set()
    for r in rallies:
        t0 = getattr(r, "t_start", None)
        t1 = getattr(r, "t_end", None)
        if t0 is None or t1 is None:
            shots = getattr(r, "shots", [])
            t0 = shots[0].t if shots else 0.0
            t1 = shots[-1].t_land if shots else 0.0
        inside = [p for p in got if t0 - 0.75 <= p.t_end <= t1 + 1.5]
        if not inside:
            continue
        recalled += 1
        p = inside[-1]
        used.add(id(p))
        if r.expected_winner and p.winner == r.expected_winner:
            winner_ok += 1
        if r.cause and p.cause == r.cause:
            cause_ok += 1
        if r.expected_score and p.score_after and p.score_after.display == r.expected_score:
            score_ok += 1

    return {
        "rallies_expected": n,
        "points_detected": len(got),
        "spurious_points": max(0, len(got) - recalled),
        "point_recall": round(recalled / max(1, n), 4),
        "winner_accuracy": round(winner_ok / max(1, recalled), 4),
        "cause_accuracy": round(cause_ok / max(1, recalled), 4),
        "score_display_accuracy": round(score_ok / max(1, recalled), 4),
        "_winner_ok": winner_ok,
    }


def _run_rules_layer(sport: str, script, mode: str = "singles") -> Dict[str, Any]:
    rules = get_rules(sport)
    court = CourtModel(script.spec, np.eye(3))
    scorer = rules.new_match(list(script.players), mode=mode, best_of=3)
    engine = make_rally_engine(sport, rules, scorer, court, mode, {})
    engine.process(perfect_events(script))
    engine.finalize_rally()
    out = _compare(engine.points, script.rallies)
    out["final_score"] = scorer.display()
    out["winner"] = scorer.winner
    return out


def _run_event_layer(sport: str, script, camera, court_cfg: dict,
                     jitter_px: float = 0.0, dropout_prob: float = 0.0,
                     seed: int = 0, mode: str = "singles") -> Dict[str, Any]:
    from ..perception.calibration import build_court

    court, _ = build_court(court_cfg, script.spec)
    track = oracle_track(script, camera, jitter_px=jitter_px,
                         dropout_prob=dropout_prob, seed=seed)
    for tp in track:
        tp.court_xy = court.to_court_point(tp.xy)
    det = EventDetector(court, fps=script.fps,
                        cfg={"sport": sport,
                             "events": {"min_bounce_prominence_px":
                                        (8.0 if sport == "badminton" else 6.0)}},
                        classifier=None)
    events = det.detect(track, n_frames=int(script.duration * script.fps)).events

    rules = get_rules(sport)
    scorer = rules.new_match(list(script.players), mode=mode, best_of=3)
    engine = make_rally_engine(sport, rules, scorer, court, mode, {})
    engine.process(events)
    engine.finalize_rally()
    out = _compare(engine.points, script.rallies)
    out["final_score"] = scorer.display()
    out["track_frames"] = len(track)
    out["contacts"] = det.detect(track).stats if False else len(events)
    return out


def _run_end_to_end(sport: str, script, camera, video_path: str,
                    mode: str = "singles") -> Dict[str, Any]:
    from ..pipeline.analyzer import MatchAnalyzer

    corners = [list(camera.project(p)) for p in
               [(0, 0), (0, script.spec.width),
                (script.spec.length, script.spec.width), (script.spec.length, 0)]]
    cfg = {
        "sport": sport,
        "evidence": {"enabled": False},
        "camera": {"court_corners_img": corners},
        "detector": {"ball": "classical"},
    }
    analyzer = MatchAnalyzer.from_config(None, cfg)
    result = analyzer.run(video_path)
    out = _compare(result.points, script.rallies)
    out["final_score"] = result.final_score
    out["detection_rate"] = result.stats.get("detection_rate")
    out["track_coverage"] = result.stats.get("track_coverage")
    return out


def run_benchmark(sport: str = "all", fps: float = 30.0, n_rallies: int = 12,
                  out_dir: Optional[str] = None, keep_video: bool = False,
                  run_end_to_end_benchmark: bool = True) -> Dict[str, Any]:
    """Benchmark one or all sports and return a JSON-serialisable report."""
    sports = ["badminton", "tennis"] if sport == "all" else [sport]
    report: Dict[str, Any] = {"fps": fps, "n_rallies": n_rallies,
                              "sports": {}, "generated_at": None}
    import time

    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for sp in sports:
        builder = build_badminton_demo if sp == "badminton" else build_tennis_demo
        script = builder(fps=fps, n_rallies=n_rallies)
        camera = SceneRenderer(script).cam
        entry: Dict[str, Any] = {}

        entry["rules"] = _run_rules_layer(sp, script)
        entry["events_clean"] = _run_event_layer(
            sp, script, camera,
            {"court_corners_img": [list(camera.project(p)) for p in
                                   [(0, 0), (0, script.spec.width),
                                    (script.spec.length, script.spec.width),
                                    (script.spec.length, 0)]]})
        entry["events_noisy"] = _run_event_layer(
            sp, script, camera,
            {"court_corners_img": [list(camera.project(p)) for p in
                                   [(0, 0), (0, script.spec.width),
                                    (script.spec.length, script.spec.width),
                                    (script.spec.length, 0)]]},
            jitter_px=3.0, dropout_prob=0.12, seed=1)

        if run_end_to_end_benchmark:
            tmpdir = out_dir or tempfile.mkdtemp(prefix="smartgym_bench_")
            os.makedirs(tmpdir, exist_ok=True)
            video_path = os.path.join(tmpdir, f"bench_{sp}.mp4")
            if not os.path.exists(video_path):
                SceneRenderer(script).write_video(video_path)
            write_ground_truth(script, os.path.join(tmpdir, f"bench_{sp}_gt.json"))
            entry["end_to_end"] = _run_end_to_end(sp, script, camera, video_path)
            if not keep_video:
                try:
                    os.remove(video_path)
                except OSError:
                    pass

        report["sports"][sp] = entry

    # headline numbers
    flat = []
    for sp, entry in report["sports"].items():
        for layer, value in entry.items():
            flat.append({"sport": sp, "layer": layer,
                         "winner_accuracy": value.get("winner_accuracy"),
                         "point_recall": value.get("point_recall")})
    report["summary"] = flat
    return report
