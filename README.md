# SmartGym — IoT/Edge-AI match scoring for badminton & tennis

An end-to-end computer-vision system that watches a match video, tracks the
ball/shuttle, decides **in/out** and the **point winner**, keeps the **official
score** for the sport, and stores **image evidence** for every decision it makes.

This repository is the ML/perception half of the *IoT Multipurpose Gymnasium*
course project. Badminton and tennis are complete (rules, scoring, evidence);
volleyball and basketball share the same perception and evidence skeleton.

```
video ──► calibrate court ──► track ball ──► extract events ──► rules engine ──► score
   (pixels)     (metres)      (trajectory)   (bounces, hits)    (BWF / ITF)      │
                                                                                ├─► JSON / CSV timeline
                                                                                └─► proof images + scoreboard
```

---

## 1. What it actually does

| Capability | Status | Where |
|---|---|---|
| Court calibration (4 corners, homography, or **auto line detection**) | ✅ | `smartgym/perception/calibration.py` |
| Ball/shuttle detection (classical CV **and** YOLO adapter) | ✅ | `smartgym/perception/ball.py` |
| Multi-frame tracking (online Kalman + **offline track-before-detect**) | ✅ | `smartgym/core/kalman.py`, `perception/linker.py` |
| Physical event extraction (bounces, racket hits, net faults, ball at rest) | ✅ | `smartgym/events/` |
| **Official badminton scoring** (21, 2-point lead, 30 cap, serve courts, change of ends) | ✅ | `smartgym/sports/badminton.py` |
| **Official tennis scoring** (15/30/40, deuce/ad, sets, tie-break, serve rotation) | ✅ | `smartgym/sports/tennis.py` |
| Rally interpretation (serve faults, double faults, double bounces, own-court) | ✅ | `smartgym/analysis/rally.py` |
| Line calls with ball-radius tolerance + `LINE_CALL` flag | ✅ | `court/model.py` |
| **Proof images** per point (decisive frame + zoom + trail + scoreboard) | ✅ | `smartgym/evidence/` |
| JSON / CSV timeline with score-before→after per point | ✅ | `smartgym/evidence/recorder.py` |
| Learned contact classifier (NumPy MLP) + dataset exporter | ✅ | `smartgym/training/` |
| Simulation harness with ground truth + layered benchmark | ✅ | `smartgym/sim/`, `training/benchmark.py` |
| Review dashboard (zero dependencies) | ✅ | `smartgym/service/dashboard.py` |
| Live streaming mode | 🟡 works, latency untuned | `tracking.mode: online` |
| Volleyball / basketball scoring | 🟡 court + detector scaffold only | `docs/ROADMAP.md` |

---

## 2. Quick start

```bash
git clone https://github.com/umeshmangalekar3421/IoT-Multipurpose_Gymnasium.git
cd IoT-Multipurpose_Gymnasium
pip install -r requirements.txt          # numpy, opencv-python-headless, pyyaml, pytest
python -m smartgym.cli info              # environment check: versions + fixes
python -m pytest tests -q                # 70 tests, no data needed
python -m smartgym.cli demo --sport badminton --out runs/demo
python -m smartgym.service.dashboard runs/demo    # http://localhost:8080
```

`pip install -e .` also works and gives you a `smartgym` command on the PATH
(`smartgym demo`, `smartgym analyze`, `smartgym info`, ...).

> **`ImportError: libGL.so.1`?** You have the GUI OpenCV build. This project
> never opens a window, so use the headless build:
> `pip uninstall -y opencv-python && pip install opencv-python-headless`.
> `python -m smartgym.cli info` diagnoses this and every other dependency.

`make demo` needs **no data at all**: it renders a synthetic badminton match with
known ground truth, analyses it, and prints a comparison:

```
=== SMARTGYM MATCH REPORT ===
points       : 12
FINAL SCORE  : 6-6
processing   : 606 fps (6.5 s total)

--- ground truth comparison ---
expected score : 6-6
detected score : 6-6
rally winners  : 12/12 matched
points scored  : 12/12
```

> **Use 60 fps.** A shuttle moves 15-25 px/frame at 30 fps, which is marginal for
> the physics layer; at 60 fps the flight is resolved properly and event
> extraction becomes reliable. Both the demo and the benchmark default to 60 fps
> for this reason.

### Analysing your own video

```bash
# 1. find the court corners (once per camera position)
python -m smartgym.cli calibrate my_match.mp4 --sport badminton --out calib.json

# 2. analyse
python -m smartgym.cli analyze my_match.mp4 \
    --config configs/badminton.yaml \
    --players "Umesh,Opponent" \
    --corners 420:100,860:100,1260:684,20:684 \
    --out runs/match1 --json runs/match1/match.json

# 3. review the evidence
python -m smartgym.service.dashboard runs/match1
```

Corner order is **clockwise from the far-left corner of the image**
(top-left → top-right → bottom-right → bottom-left). `calibrate` prints exactly
that order for you.

---

## 3. What a decision looks like

Every point produces a JSON record with full provenance:

```json
{
  "index": 5,
  "t_start": 28.4, "t_end": 30.09,
  "frame_start": 852, "frame_end": 903,
  "winner": "P1", "loser": "P2",
  "cause": "landing_out",
  "confidence": 0.42,
  "score_before": {"display": "2-2"},
  "score_after":  {"display": "3-2"},
  "events": [
    {"type": "serve",       "t": 28.57, "court_xy": [4.10, 3.02]},
    {"type": "bounce_in",   "t": 29.10, "court_xy": [7.90, 1.90], "in_bounds": true},
    {"type": "landing_out", "t": 30.09, "court_xy": [13.95, 2.98], "in_bounds": false}
  ],
  "evidence": ["evidence/point_005_landing.jpg",
               "evidence/point_005_trail.jpg",
               "evidence/point_005_scoreboard.jpg"]
}
```

The proof image contains the decisive frame, the flight trail, a zoomed view of
the landing point (`45 cm from line`), the rule that was applied and the score
before → after:

![proof image layout](docs/proof_layout.txt)

---

## 4. Architecture

```
smartgym/
├── core/            geometry (homography, curves), Kalman tracker, config, types
├── court/           court specifications + CourtModel (pixel ⇄ metre, line calls)
├── perception/      video, calibration, ball detectors, player tracker, linker
├── events/          trajectory physics → semantic events (bounce/hit/net/lost)
├── analysis/        rally state machines (serve faults, double bounces, decisions)
├── sports/          official rules: landing adjudication + scoring bookkeeping
├── evidence/        annotated proof images, scoreboards, proof sheets
├── pipeline/        the orchestrator: video in → score + evidence out
├── training/        features, NumPy MLP, dataset export, layered benchmark
├── sim/             renderer + scripted matches with verified ground truth
├── service/         review dashboard (stdlib HTTP server)
└── cli.py           analyze / calibrate / demo / benchmark / train / export
```

Two deliberate design decisions are worth defending in a viva:

**1. Geometry is separated from bookkeeping.** `sports/*` contains the rule books
(`SportRules` = in/out adjudication, `MatchScorer` = official scoring). Neither
knows anything about video. That is why the scoring layer is exhaustively
unit-tested (43 tests) and reaches **100 %** accuracy on the benchmark, while the
perception layer can be swapped or retrained without touching a single rule.

**2. Detection confidence never becomes a scored point without provenance.**
Every point carries the events that produced it, the score before and after, and
an image. A referee disagreeing with a call has everything needed to overturn it
— and the system marks genuinely uncertain decisions with `needs_review`.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the data flow and
[`docs/RULES.md`](docs/RULES.md) for the exact laws/rules implemented.

---

## 5. Honest accuracy report

`make benchmark` measures three layers separately against simulated ground truth
so that failures can be attributed:

| Layer | What it isolates | Badminton | Tennis |
|---|---|---|---|
| `rules` (perfect events) | rule-book bookkeeping | **100 %** winner, 100 % points, 100 % cause | **100 %** winner, 100 % points, 100 % cause |
| `events_clean` (perfect track, 60 fps) | physics heuristics | **100 %** winner, 100 % points | 25-65 % winner (seed dependent), 100 % points |
| `events_noisy` (3 px jitter + 12 % dropout) | robustness | ~100 % winner, 1 spurious point | ~25-40 % winner, 2 spurious points |
| `end_to_end` (rendered video, classical CV) | full perception stack | see `runs/benchmark/benchmark_all.json` | see `runs/benchmark/benchmark_all.json` |

**Interpretation (important, and deliberately not hidden):**

* The rules and scoring layer is exact — it is a direct transcription of the BWF
  laws and ITF rules and is tested against them.
* The *event* layer (deciding "this contact was a bounce, that one was a
  stroke") is the current bottleneck. It is classical physics with thresholds,
  not a learned model. It works reliably when the ball is tracked continuously
  and degrades when the ball leaves the frame. Badminton (few, decisive floor
  contacts) is solved; tennis (many harmless bounces, two-serve bookkeeping)
  still loses rallies when a bounce is missed.
* On **real footage** the classical detector is a bootstrap: motion-based
  detection needs the ball to be the only small moving object. The repo ships
  the complete path to a trained detector (`train-ball`, `export-dataset`,
  `YoloBallDetector`) and a learned contact classifier, which is what a
  production deployment should use.

This split is intentional: it makes the improvement path measurable instead of
hand-wavy, and it means the project still *scores matches correctly* when given
a good ball track (the `--perception oracle` demo proves this end to end).

---

## 6. The learned model

Two learnable components, both trainable on a laptop CPU:

```bash
# 1. contact classifier (bounce vs stroke vs noise) — 16 physics features
python -m smartgym.cli export-dataset my_match.mp4 --sport tennis --out data
python -m smartgym.cli train-events data/events_my_match.mp4.csv --out models/event_mlp.npz

# 2. ball detector (YOLO) — bootstrap labels from the classical detector
python -m smartgym.cli train-ball clip1.mp4,clip2.mp4 --sport badminton --out data/ball
yolo detect train data=data/ball/data.yaml model=yolov8n.pt epochs=100 imgsz=640
# then: detector.ball: yolo, detector.weights: runs/detect/train/weights/best.pt
```

The classifier is blended 60/40 with the physics confidence rather than replacing
it, so a badly calibrated model degrades gracefully. Details and the feature
table are in [`docs/TRAINING.md`](docs/TRAINING.md).

---

## 7. Edge / IoT deployment

The heavy lifting is OpenCV + NumPy, so the same code runs on:

| Target | How |
|---|---|
| Laptop / lab PC | `python -m smartgym.cli analyze ...` (≈500 fps on a 2-core VM for scoring; CV is the cost) |
| NVIDIA Jetson Nano/Orin | `pip install opencv-python-headless`, export the detector to ONNX/TensorRT, `detector.ball: yolo` |
| Raspberry Pi 4/5 + camera | `tracking.mode: online` + a small YOLO-nano model; the rules layer is pure Python and cheap |
| Phone / browser | the rules layer is NumPy-only, so it can be ported to Swift/Kotlin or run in Pyodide |

The dashboard's `/healthz`, `/api/match` and `/api/points` endpoints are the
integration surface for a gym display or a club app. Recommended wiring:
camera → edge box (this repo) → MQTT/HTTP → scoreboard + dashboard.

---

## 8. Repository map

```
configs/            per-sport YAML (every key documented)
docs/               architecture, rules, training, project report, roadmap
smartgym/           the library (see §4)
tests/              70 tests: scoring laws, geometry, tracking, events, evidence
Makefile            install / test / demo / benchmark / dashboard
```

## 9. Limitations (read before trusting a score)

1. **Height is not observed.** A single camera cannot measure the ball's height,
   so landing points are inferred from the lowest point of the trajectory and
   from ballistic extrapolation. Near the baseline this is accurate to a few
   centimetres; for a ball lost high in the air it is an estimate, and the point
   is flagged `needs_review`.
2. **Occlusion.** If the ball disappears for more than a few frames the rally is
   reconstructed from the rule engine's expectations, not from evidence.
3. **Camera.** Fixed, elevated, whole-court view. Panning cameras need
   stabilisation (not implemented).
4. **Lighting.** The classical detector assumes a reasonably lit court. A trained
   detector removes this assumption.
5. **Doubles** is supported by the rules (service courts, alleys) but the
   perception layer tracks only one ball and does not separate the two players
   on a side.

## 10. Licence

MIT — free for coursework and club use.
