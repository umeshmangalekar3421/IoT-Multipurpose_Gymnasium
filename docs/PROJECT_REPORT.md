# Project report — ML scoring engine for the IoT Multipurpose Gymnasium

*Course project deliverable. Everything below is reproducible with the commands
in §7; no dataset is required.*

---

## 1. Problem statement

The multipurpose gymnasium needs an automatic scoring and analysis layer that
works from **ordinary video** (a phone on a tripod), without sensors on the court
and without a human operator. The requirements taken from the project brief:

| # | Requirement | Delivered |
|---|---|---|
| R1 | Detect the ball/shuttle and its position over time | ball detector + tracker, image and court coordinates |
| R2 | Decide points: in/out, faults, rallies | `sports/` + `analysis/rally.py` |
| R3 | Keep the score according to each sport's official rules | BWF badminton + ITF tennis, 100 % on the benchmark |
| R4 | Update the score **on a timestamp** | every point carries `t_start`, `t_end`, `frame_*` |
| R5 | **Store proof in image format** | 3 annotated JPEGs per point + proof sheet |
| R6 | Work for at least two sports | badminton and tennis are complete |
| R7 | Be deployable on IoT/edge hardware | OpenCV + NumPy only; Jetson/Pi notes in the README |

Sports chosen: **badminton** (highest shuttle speeds, harshest occlusion — the
hard case) and **tennis** (clean ballistics, bounce-based rules — the easy case).
Together they cover both rule families: *rally point* scoring and *bounce-based*
scoring.

## 2. Why this is an ML / edge-AI project and not a script

Three learned or learnable components, plus a data engine:

1. **Detector** — classical motion+appearance fusion out of the box, with a
   YOLO fine-tuning path (export → label → train → deploy) for real footage.
2. **Contact classifier** — a 16-feature, 3-class MLP (NumPy implementation)
   that learns *bounce vs stroke vs noise*, replacing hand-tuned thresholds.
3. **Trajectory linking** — a velocity-aware global association that behaves
   like a small learned tracker (cost function + plausibility model).
4. **Simulation engine** — renders physically-plausible matches with exact
   ground truth so models can be trained and *measured* without a labelled
   dataset. This is the piece that makes the project iterative.

## 3. System design

Detailed in `docs/ARCHITECTURE.md`. The essential decision:

> **Geometry and bookkeeping are separated from perception.**

`sports/*` is a pure transcription of the rule books operating on court metres
and an event list. It has no video dependency, which is why it is exhaustively
testable and exact. Perception produces events; it never scores.

Consequences:
* the rules engine can be validated independently (43 tests, 100 % benchmark),
* a broken detector cannot corrupt the rules,
* the same rules engine runs on the edge box, in the cloud, or ported to a phone.

## 4. Method details worth defending

### 4.1 Court calibration
Three tiers: clicked corners (accurate), an offline homography (reproducible), or
automatic line detection. The automatic path finds the painted-line network as
connected components, approximates the court quadrilateral, then **disambiguates
the four rotations and two mirrorings** by measuring how well the *inner* court
lines (service lines, centre line) land on real line pixels. That disambiguation
is what makes auto-calibration usable: a court rectangle looks the same rotated
90°, but only one interpretation puts the service lines where they are painted.

### 4.2 In/out judgement
Landing points are computed in court metres through the homography. The verdict
uses the **ball radius tolerance** (`line_margin_m`): a ball whose centre is 3 cm
outside a line is still IN, because that is how a line judge rules and it is the
width of the painted line. Landings within `margin + 2 cm` are additionally
flagged `LINE_CALL` for review.

### 4.3 Physical event signatures
| Event | Signature | Space |
|---|---|---|
| bounce | local maximum of image-y + direction reversal + speed loss | image |
| stroke | sign reversal of image-x velocity | image |
| ball at rest | speed below threshold for N frames | either |
| net contact | arc approaches the net plane and returns | court |

Image space is used deliberately: projecting an **airborne** ball through a
**floor** homography displaces it by an amount that varies with height, so the
"court trajectory" of a straight flight swings wildly. This was measured during
development and is the main reason the naive approach fails.

### 4.4 Ballistic landing extrapolation
When the ball is lost mid-flight, a parabola is fitted to the last samples and
its vertex is taken as the floor contact. Without this, an in/out call on a ball
that leaves the top of the frame is pure guesswork.

### 4.5 Rally state machine
A pending decision + a confirmation window prevents double counting: an in-bounds
landing is held for ~10 frames, and a racket contact inside the window revokes
it. A `terminal` flag on bounces handles the case where the ball is returned
after touching the floor.

### 4.6 Evidence
Every point stores the decisive frame with overlay, a zoomed landing, the flight
trail, the applied rule, the score before → after, and a rendered scoreboard.
Decisions with low confidence are flagged `needs_review`.

## 5. Results

Measured with `python -m smartgym.cli benchmark --sport all`.

| Layer | Badminton | Tennis |
|---|---|---|
| Rules on perfect events | **100 %** winner, 100 % points, 100 % cause | **100 %** winner, 100 % points, 100 % cause |
| Events, perfect track (60 fps) | **100 %** winner, 100 % points | 25-65 % winner, 100 % points |
| Events, jitter + 12 % dropout | ~100 % winner, 1 spurious point | ~25-40 % winner, 2 spurious points |
| Demo (rendered video, oracle ball track) | **score exact 6-6, 12/12 rally winners** | see `runs/demo_tennis` |

Test suite: **70 tests, 0.4 s**, covering every implemented law/rule, the
geometry, the tracker, the event detector, the rally engines and the evidence
writer.

## 6. Limitations and future work

1. **Event layer is classical.** It is the current bottleneck and the clearest
   route to improvement: train the contact classifier on real, hand-checked
   contacts (the pipeline for this is already shipped).
2. **Height is unobserved.** Landing points near the baseline are accurate to a
   few centimetres; a ball lost high in the air is an estimate (flagged).
3. **Detector needs motion.** Fine for a tripod camera, weak for handheld.
   A trained detector removes this.
4. **No player identification.** Sides are known; identities are not
   (`detector.players` adds boxes usable for strike attribution).
5. **Doubles** rules are implemented; the perception layer does not yet model
   two players per side (formation tracking is the next step — see
   `docs/ROADMAP.md`).

## 7. Reproducibility

```bash
pip install -r requirements.txt

make test        # 70 unit tests, no data needed
make demo        # render + score a full match, write proof images
make benchmark   # layered accuracy report
make dashboard   # browse the result / evidence

# real footage
python -m smartgym.cli calibrate my.mp4 --out calib.json
python -m smartgym.cli analyze my.mp4 --config configs/badminton.yaml \
       --corners 420:100,860:100,1260:684,20:684 --out runs/match1
```

Deterministic: the simulation uses seeded RNGs, so every number in §5 is
reproducible bit-for-bit on the same machine.

## 8. What was learned (engineering reflections)

* **Separating adjudication from perception was the single most valuable
  decision.** It produced an exact, testable, defensible core and a clearly
  bounded, improvable perception layer.
* **Measuring in layers** turned "the accuracy is bad" into "the event layer is
  the bottleneck", which is actionable.
* **Physical reasoning beats threshold tuning.** Every improvement that survived
  (image-space bounces, velocity-aware linking, striker-from-landing-side,
  terminal bounces, ballistic extrapolation) came from asking what the ball
  *must* do, not from fitting a constant.
* **A simulator is a dataset.** It provided ground truth for a project with no
  labelled video, and it doubles as an offline demo.
