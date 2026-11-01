# Architecture and data flow

## 1. Stages

```
                    ┌──────────────────────────────────────────────┐
 clip.mp4 ─────────►│ 1. probe       fps, size, duration           │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │ 2. calibrate   image ⇄ court metres (3 ways) │
                    └───────────────┬──────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ 3. perceive                                                       │
   │    ball detector → candidates → global linker → track             │
   │    (optional) player tracker → side attribution                   │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ 4. events   segments → bounces (y-maxima) → hits (x-reversal)     │
   │             → net contacts → ball-at-rest → ballistic landings    │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ 5. rules    RallyEngine (sport-specific) + MatchScorer (official) │
   │             → PointRecord[] with score_before/after               │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ 6. evidence second video pass → proof JPEGs, scoreboard, timeline │
   └───────────────────────────────────────────────────────────────────┘
```

Passes over the video: **2** by default (perception, evidence) and **3** if
`output.annotated_video` is set. Nothing is buffered in RAM beyond candidate
coordinates, so multi-hour matches fit in a few hundred MB.

## 2. Key data structures

| Type | Meaning | Module |
|---|---|---|
| `Detection` | one ball candidate in image pixels | `core/types.py` |
| `TrackPoint` | ball position at a frame, image + court metres | `core/types.py` |
| `Segment` | one continuous flight arc | `events/trajectory.py` |
| `Contact` | bounce / hit / net / lost, with sub-frame refinement | `events/trajectory.py` |
| `Event` | semantic, adjudicable observation | `core/types.py` |
| `LandingVerdict` | in/out + zone + distance to line | `sports/base.py` |
| `PointRecord` | a scored rally + both scoreboards + evidence paths | `core/types.py` |
| `ScoreState` | immutable scoreboard snapshot | `core/types.py` |

Every one of them has `as_dict()` producing JSON-safe primitives (numpy scalars
are converted), so the whole match can be streamed to an IoT gateway.

## 3. Why offline linking beats online tracking

The default `tracking.mode: offline` collects *all* ball candidates first, then
links them:

```
cost(track, candidate) = |candidate − prediction(track)| − 45·confidence
accept if |candidate − prediction| ≤ gate AND implied speed ≤ physical limit
```

A constant-velocity prediction is what makes this robust: a ball in flight is
predictable, a flickering line or a spectator is not. Then tracks are scored by
length, confidence, travel and duration, and physically implausible ones
(too short, parked in one place) are discarded.

`tracking.mode: online` runs the same idea frame by frame with a Kalman filter,
which is what a live RTSP/CSI stream needs. It has extra distractor rejection
(stationary-track detection, speed-aware gating) that the offline mode does not
need.

## 4. The physical signatures used

| Question | Signal | Space | Why image space |
|---|---|---|---|
| Did the ball touch the floor? | local **maximum of y**, with a direction reversal and a speed drop | image | the floor-projected trajectory of an airborne ball swings wildly because height changes its projection |
| Was it struck by a racket? | sign reversal of **vx** | image | same reason; a stroke sends the ball back horizontally |
| Did the ball stop? | speed < threshold for N frames | either | a resting ball is the end of a rally |
| Did it hit the net? | the arc approaches the net plane and comes back | court | net position is a court landmark |
| Where will a lost ball land? | parabola through the last ~15 samples → vertex | image → court | constant-gravity motion is a parabola in the image too |

## 5. Rule-engine state machine

`RallyEngine` is intentionally tiny: it holds `in_rally`, `striker_side`,
`serve_pending` (badminton) / `faults` (tennis), `expecting_serve`, and at most
one `PendingDecision`. Two mechanisms prevent double-counting:

* **Confirmation window** (`confirm_landing_frames`): an in-bounds landing is
  held for N frames; if a racket contact arrives inside the window the landing is
  discarded as a false bounce. Out-of-bounds landings are never re-opened.
* **Terminal flag**: a floor contact after which the ball demonstrably keeps
  flying is informational only. This is what stops a single rally being paid out
  as two points.

The window is evaluated *before* the next event is interpreted, so a
late-arriving event belongs to the next rally rather than the finished one.

## 6. Evidence design

Each point writes three JPEGs:

| File | Contents |
|---|---|
| `point_NNN_landing.jpg` | decisive frame, court overlay, flight trail, ring + crosshair on the landing, zoom inset, text panel with point/time/confidence/winner/cause/score-before→after, and the last events with court coordinates |
| `point_NNN_trail.jpg` | the whole rally trajectory on one frame |
| `point_NNN_scoreboard.jpg` | rendered scoreboard card |

Plus `match_timeline.jpg` (a proof sheet of every point), `match.json`,
`score_timeline.json` and `score_timeline.csv`.

Everything is drawn with OpenCV primitives so it works headless on an edge
device; no fonts or browser needed.

## 7. Testing strategy

| Layer | Test file | What is asserted |
|---|---|---|
| Rules | `tests/test_scoring.py` | every law: 21-point games, 20-all, 30 cap, serve parity, change of ends, deuce/ad, tie-break rotation, no-ad |
| Geometry | `tests/test_court_and_rules.py` | homography round-trip, corner convention, in/out with ball radius, service boxes, singles/doubles geometry |
| Perception | `tests/test_pipeline.py` | tracker coasting + distractor rejection, linker consistency, bounce localisation, segment splitting |
| Rules pipeline | `tests/test_pipeline.py` | landing in/out, net fault, serve fault, double fault, double bounce, non-terminal bounce, provenance |
| Evidence | `tests/test_pipeline.py` | images/CSV/JSON are written and readable |

70 tests, no network, no data files, ~0.4 s.

## 8. Extension points

* **New sport**: implement `SportRules` + `MatchScorer` + a `RallyEngine`
  subclass, register in `sports/registry.py` and `analysis/rally.py`.
* **New detector**: subclass `BaseBallDetector` (`initialize`, `detect`).
* **Better event model**: implement a `predict_proba(X)` object and pass it as
  `EventDetector(classifier=...)`; `training/event_model.py` provides one.
* **New evidence style**: `evidence/annotate.py` is a set of pure drawing
  functions.
* **Live transport**: `PerceptionPipeline` accepts any `VideoSource`; wrap a
  GStreamer/RTSP reader with the same `frames()` / `read_window()` interface.
