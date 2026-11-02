"""Command line interface.

    python -m smartgym.cli analyze  clip.mp4 --config configs/badminton.yaml
    python -m smartgym.cli demo     --sport badminton --out runs/demo
    python -m smartgym.cli calibrate clip.mp4 --out calib.json
    python -m smartgym.cli benchmark --sport badminton
    python -m smartgym.cli train-events data/events.csv --out models/event_mlp.pt

The ``demo`` command renders a synthetic clip with known ground truth, analyses
it and prints a comparison — a self-contained validation that needs no data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .core.config import load_config
from .core.types import Sport
from .court.model import PRESETS
from .sports.registry import available_sports


def _print_result(result, verbose: bool = False) -> None:
    print("\n=== SMARTGYM MATCH REPORT ===")
    meta = result.meta
    print(f"sport        : {meta.get('sport')} ({meta.get('format')})")
    print(f"players      : {meta.get('players')}")
    print(f"video        : {meta.get('video')}  {meta.get('frames')} frames "
          f"@ {meta.get('fps')} fps  ({meta.get('duration_s')} s)")
    cal = result.calibration or {}
    print(f"calibration  : {cal.get('method')} "
          f"(confidence {cal.get('confidence', 'n/a')})")
    print(f"processing   : {meta.get('processing_fps')} fps "
          f"({meta.get('elapsed_s')} s total)")
    print(f"points       : {len(result.points)}")
    print(f"FINAL SCORE  : {result.final_score}")
    print(f"WINNER       : {result.winner or 'undecided'}")
    if verbose:
        print("\n#  t(s)   winner  cause              score after")
        for p in result.points:
            flag = " [review]" if p.needs_review else ""
            print(f"{p.index:<3}{p.t_end:<7.2f}{p.winner:<8}{p.cause:<19}"
                  f"{(p.score_after.display if p.score_after else ''):<18}{flag}")
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings[:12]:
            print(f"  - {w}")
    if result.artifacts:
        print("\nartifacts:")
        for k, v in result.artifacts.items():
            print(f"  {k:<16} {v}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline.analyzer import MatchAnalyzer

    overrides: Dict[str, Any] = {}
    if args.sport:
        overrides["sport"] = args.sport
    if args.players:
        overrides.setdefault("match", {})["players"] = args.players.split(",")
    if args.corners:
        pts = [tuple(float(v) for v in c.split(":")) for c in args.corners.split(",")]
        overrides.setdefault("camera", {})["court_corners_img"] = [list(p) for p in pts]
    if args.detector:
        overrides.setdefault("detector", {})["ball"] = args.detector
    if args.weights:
        overrides.setdefault("detector", {})["weights"] = args.weights
    if args.format:
        overrides.setdefault("match", {})["format"] = args.format
    if args.max_frames:
        overrides.setdefault("video", {})["max_frames"] = args.max_frames
    if args.stride:
        overrides.setdefault("video", {})["stride"] = args.stride
    if args.annotated_video:
        overrides.setdefault("output", {})["annotated_video"] = args.annotated_video
    if args.out:
        overrides.setdefault("evidence", {})["out_dir"] = args.out

    analyzer = MatchAnalyzer.from_config(args.config, overrides)
    result = analyzer.run(args.video)
    _print_result(result, verbose=not args.quiet)
    if args.json:
        result.save(args.json)
        print(f"\nmatch json written to {args.json}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Probe a clip and propose court corners (auto-detected)."""
    from .perception.calibration import AutoCalibrator, validate_homography
    from .perception.video import VideoSource

    sport = (args.sport or "badminton").lower()
    spec = PRESETS[sport]
    with VideoSource(args.video) as video:
        frames = [f for i, f in video.read_window(0, 120) if i % 20 == 0][:5]
        first = frames[0] if frames else None
        if first is None:
            print("could not read frames", file=sys.stderr)
            return 2
        cal = AutoCalibrator(spec).calibrate_over_frames(frames)
        if cal is None:
            print("auto-calibration failed: no court lines detected.")
            print("Provide manual corners, e.g.\n"
                  "  --corners 210:150,1690:150,1900:960,20:960")
            return 1
        ok, why = validate_homography(cal.homography, spec,
                                      (first.shape[1], first.shape[0]))
        payload = {
            "sport": sport,
            "method": cal.method,
            "confidence": round(cal.confidence, 3),
            "valid": ok,
            "note": why,
            "court_corners_img": [[round(x, 2), round(y, 2)] for x, y in cal.corners],
        }
        print(json.dumps(payload, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"\nwritten to {args.out}")
        return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Render a synthetic match, analyse it, write evidence, compare to truth."""
    from .sim.scripts import build_demo
    from .sim.scene import SceneRenderer, oracle_track, write_ground_truth
    from .pipeline.analyzer import MatchAnalyzer

    os.makedirs(args.out, exist_ok=True)
    script = build_demo(args.sport, fps=args.fps, n_rallies=args.rallies)
    renderer = SceneRenderer(script, noise=args.stress, player_sway=args.stress)
    camera = renderer.cam
    video_path = os.path.join(args.out, f"demo_{args.sport}.mp4")
    gt_path = os.path.join(args.out, f"demo_{args.sport}_ground_truth.json")
    if not (os.path.exists(video_path) and args.cached):
        print(f"rendering {script.duration:.1f}s of {args.sport} at {args.fps} fps ...")
        renderer.write_video(video_path)
    write_ground_truth(script, gt_path)
    print(f"video      : {video_path}")
    print(f"groundtruth: {gt_path}")

    corners = [list(camera.project(p)) for p in
               [(0, 0), (0, script.spec.width),
                (script.spec.length, script.spec.width), (script.spec.length, 0)]]
    overrides = {
        "sport": args.sport,
        "evidence": {"out_dir": args.out},
        "camera": {"court_corners_img": corners},
        "detector": {"ball": "classical", "players": "none"},
    }
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", f"{args.sport}.yaml")
    analyzer = MatchAnalyzer.from_config(
        cfg_path if os.path.exists(cfg_path) else None, overrides)

    track = oracle_track(script, camera) if args.perception == "oracle" else None
    if args.perception == "oracle":
        print("perception : ORACLE (known ball track; validates event/rules/evidence)")
    else:
        print("perception : classical CV (detector + linkers; measures the CV baseline)")

    result = analyzer.run(video_path, oracle_track=track)
    _print_result(result, verbose=True)

    ref = script.reference
    print("\n--- ground truth comparison -------------------------------")
    print(f"expected score : {ref['final_score']}")
    print(f"detected score : {result.final_score}")
    matched = sum(1 for p, r in zip(result.points, script.rallies)
                  if r.expected_winner and p.winner == r.expected_winner)
    print(f"rally winners  : {matched}/{ref['rallies']} matched")
    print(f"points scored  : {len(result.points)}/{ref['rallies']}")
    print("\nreview dashboard:")
    print(f"  python -m smartgym.service.dashboard {args.out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .training.benchmark import run_benchmark

    report = run_benchmark(sport=args.sport, fps=args.fps,
                           n_rallies=args.rallies, out_dir=args.out,
                           keep_video=args.keep,
                           run_end_to_end_benchmark=not args.no_end_to_end)
    # a plain-text table is far easier to read (and to quote in a report) than JSON
    print(f"\n{'sport':<10}{'layer':<14}{'points':>8}{'expected':>10}"
          f"{'winner acc':>12}{'recall':>9}{'cause acc':>11}")
    print("-" * 74)
    for row in report["summary"]:
        layer = report["sports"][row["sport"]][row["layer"]]
        print(f"{row['sport']:<10}{row['layer']:<14}"
              f"{layer.get('points_detected', '')!s:>8}"
              f"{layer.get('rallies_expected', '')!s:>10}"
              f"{layer.get('winner_accuracy', 0):>12.2f}"
              f"{layer.get('point_recall', 0):>9.2f}"
              f"{layer.get('cause_accuracy', 0):>11.2f}")
    print("\nlayers: rules = official scoring on perfect events; "
          "events_* = physics from a perfect ball track;\n"
          "        end_to_end = rendered video through the full perception stack")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"benchmark_{args.sport}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nfull report: {path}")
    return 0


def cmd_train_events(args: argparse.Namespace) -> int:
    from .training.event_model import train_from_csv

    metrics = train_from_csv(args.csv, out_path=args.out, epochs=args.epochs,
                             hidden=args.hidden, seed=args.seed)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_export_dataset(args: argparse.Namespace) -> int:
    from .training.dataset import build_dataset

    stats = build_dataset(args.video, sport=args.sport, out_dir=args.out,
                          config=args.config, max_frames=args.max_frames)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train_ball(args: argparse.Namespace) -> int:
    from .training.dataset import export_yolo_dataset

    stats = export_yolo_dataset(videos=args.videos.split(","), sport=args.sport,
                                out_dir=args.out, config=args.config)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .service.dashboard import serve

    serve(args.run_dir, host=args.host, port=args.port)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print("SmartGym sports with full rules + scoring :", ", ".join(available_sports()))
    print("Court presets:")
    for name, spec in PRESETS.items():
        print(f"  {name:<10} {spec.length:.2f} x {spec.width:.2f} m "
              f"(singles {spec.singles_width:.2f} m), net {spec.net_height_m} m")
    print("\nEnvironment:")
    problems = _environment_report()
    return 1 if problems else 0


def _environment_report() -> List[str]:
    """Check the runtime and print an actionable message for each failure.

    Every entry here is a problem that *actually* stopped a run during
    development, so the checks are worth their lines: they turn a traceback into
    a one-line fix.
    """
    problems: List[str] = []
    print(f"  python      {sys.version.split()[0]}")
    try:
        import numpy as np
        print(f"  numpy       {np.__version__}")
    except Exception as exc:                                  # pragma: no cover
        print(f"  numpy       MISSING ({exc})")
        problems.append("pip install numpy")

    try:
        import cv2
        print(f"  opencv      {cv2.__version__}")
        try:
            info = cv2.getBuildInformation()
            if "GUI: " in info and "NONE" in info.split("GUI: ", 1)[1][:20]:
                print("  opencv gui  none (headless build - expected)")
        except Exception:
            pass
    except Exception as exc:                                  # pragma: no cover
        msg = str(exc)
        if "libGL" in msg:
            print("  opencv      BROKEN: the GUI build is installed but libGL is "
                  "missing (headless Linux / Docker / Jetson / Pi)")
            problems.append("pip uninstall -y opencv-python && "
                            "pip install opencv-python-headless")
        else:
            print(f"  opencv      MISSING ({msg})")
            problems.append("pip install opencv-python-headless")

    try:
        import yaml
        print(f"  pyyaml      {yaml.__version__}")
    except Exception:                                         # pragma: no cover
        print("  pyyaml      MISSING")
        problems.append("pip install pyyaml")

    try:
        import pytest
        print(f"  pytest      {pytest.__version__} (test suite available)")
    except Exception:                                         # pragma: no cover
        print("  pytest      not installed (only needed for `pytest tests`)")

    if problems:
        print("\nproblems found:")
        for p in problems:
            print(f"  -> {p}")
        print("\nfix everything at once:  pip install -r requirements.txt")
    else:
        print("\nall runtime dependencies OK")
    return problems


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smartgym",
        description="Edge-AI scoring for badminton and tennis: ball tracking, "
                    "in/out line calls, official scoring and image evidence.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse a match video")
    a.add_argument("video")
    a.add_argument("--config", help="YAML config (sport, camera, players ...)")
    a.add_argument("--sport", choices=available_sports())
    a.add_argument("--format", choices=["singles", "doubles"], dest="format")
    a.add_argument("--players", help="comma separated, e.g. P1,P2")
    a.add_argument("--corners", help="court corners in image px: x1:y1,x2:y2,x3:y3,x4:y4 "
                                     "(TL,TR,BR,BL)")
    a.add_argument("--detector", choices=["classical", "yolo", "none"])
    a.add_argument("--weights", help="model weights for the yolo detector")
    a.add_argument("--max-frames", type=int, dest="max_frames")
    a.add_argument("--stride", type=int)
    a.add_argument("--annotated-video", dest="annotated_video")
    a.add_argument("--out", help="output directory for evidence + timeline")
    a.add_argument("--json", help="write the full match JSON here")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="auto-detect court corners in a clip")
    c.add_argument("video")
    c.add_argument("--sport", choices=available_sports(), default="badminton")
    c.add_argument("--out")
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("demo", help="render + analyse a synthetic match (no data needed)")
    d.add_argument("--sport", choices=available_sports(), default="badminton")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--fps", type=float, default=60.0,
                   help="sampling rate; 60 fps is recommended (a shuttle moves "
                        "15-25 px/frame at 30 fps, which is marginal for the "
                        "physics layer)")
    d.add_argument("--rallies", type=int, default=12)
    d.add_argument("--perception", choices=["oracle", "classical"], default="oracle",
                   help="oracle = known ball track (default: validates scoring); "
                        "classical = run the real CV detector")
    d.add_argument("--stress", action="store_true",
                   help="add sensor noise and swaying players to the render")
    d.add_argument("--cached", action="store_true", help="reuse an existing demo video")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("benchmark", help="measure accuracy against simulated ground truth")
    b.add_argument("--sport", choices=available_sports() + ["all"], default="all")
    b.add_argument("--fps", type=float, default=60.0)
    b.add_argument("--rallies", type=int, default=12)
    b.add_argument("--out", default="runs/benchmark")
    b.add_argument("--keep", action="store_true", help="keep generated demo videos")
    b.add_argument("--no-end-to-end", action="store_true", dest="no_end_to_end",
                   help="skip the (slow) full-perception layer")
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train-events", help="train the bounce/hit classifier")
    t.add_argument("csv")
    t.add_argument("--out", default="models/event_mlp.npz")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--hidden", type=int, default=24)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train_events)

    e = sub.add_parser("export-dataset", help="export contact features + labels")
    e.add_argument("video")
    e.add_argument("--sport", choices=available_sports(), default="badminton")
    e.add_argument("--config")
    e.add_argument("--out", default="data")
    e.add_argument("--max-frames", type=int, dest="max_frames")
    e.set_defaults(func=cmd_export_dataset)

    y = sub.add_parser("train-ball", help="export frames for ball-detector training")
    y.add_argument("videos", help="comma separated video paths")
    y.add_argument("--sport", choices=available_sports(), default="badminton")
    y.add_argument("--config")
    y.add_argument("--out", default="data/ball")
    y.set_defaults(func=cmd_train_ball)

    i = sub.add_parser("info", help="show supported sports and court presets")
    i.set_defaults(func=cmd_info)

    srv = sub.add_parser("dashboard", help="serve the match review dashboard")
    srv.add_argument("run_dir", nargs="?", default="runs/demo",
                     help="directory containing match.json + evidence/")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8080)
    srv.set_defaults(func=cmd_dashboard)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
