# Training the models

Two components are learnable. Both can be trained on a laptop CPU in minutes.

---

## A. Contact classifier — "is this a bounce, a stroke, or noise?"

### Why it exists

`events/trajectory.py` decides bounce-vs-stroke with physics thresholds
(turn angle, speed drop, prominence). Those thresholds were tuned on one camera.
A classifier replaces them with a model that can be retrained per venue, per
sport, and per lighting condition.

The detector *blends* the model with the physics:

```
confidence = 0.6 · physics_confidence + 0.4 · P(bounce | features)
```

so a badly calibrated model degrades gracefully instead of destroying the match.

### Features (16, all physical and scale-free)

| # | Feature | Rationale |
|---|---|---|
| 0 | speed before contact (÷ image diagonal) | fast objects behave differently |
| 1 | speed after contact | |
| 2 | speed ratio after/before | a bounce loses energy; a stroke adds it |
| 3 | vy before | downward motion |
| 4 | vy after | a bounce flips vy |
| 5 | vx before (signed) | direction of travel |
| 6 | vx after (signed) | a stroke flips vx |
| 7 | turn angle (deg / 180) | strokes turn sharply, bounces do not |
| 8 | is local y maximum | the bounce signature |
| 9 | is local y minimum | an apex, where nothing happened |
| 10 | distance to nearest painted line (m) | line flicker produces false contacts on lines |
| 11 | distance to net (m) | net contacts are special |
| 12 | y relative to the segment range | height proxy |
| 13 | detector confidence | |
| 14 | frames since the previous contact | rallies have a rhythm |
| 15 | near half / far half | perspective effects differ |

### Data

```bash
# bootstrap from simulated matches (no annotation needed)
python -m smartgym.cli export-dataset my_match.mp4 --sport badminton --out data
python -m smartgym.training.dataset   # or use the API below
```

```python
from smartgym.training.dataset import build_simulated_dataset, build_dataset

build_simulated_dataset(n_per_sport=3, out_csv="data/events_sim.csv")
build_dataset("clip.mp4", sport="tennis", out_dir="data")   # real footage
```

Labels are **semi-automatic**: contacts produced by the physics are used as weak
labels (`bounce` / `hit` / `other`), and any contact that the rule engine used to
award a point is a confirmed bounce. Reviewing a handful of contested points in
the dashboard and correcting them is the highest-value labelling work.

### Train

```bash
python -m smartgym.cli train-events data/events_sim.csv \
       --out models/event_mlp.npz --epochs 300 --hidden 24
```

```python
from smartgym.training.event_model import load_csv, train_from_csv
metrics = train_from_csv("data/events_sim.csv", "models/event_mlp.npz")
print(metrics["val_accuracy"], metrics["confusion"])
```

The model is a 16→24→12→3 MLP trained with Adam + class-weighted cross-entropy,
implemented in NumPy (no PyTorch). It saves as `.npz` (compact) or `.json`
(human readable), and `export_onnx_weights()` dumps a framework-neutral
description for ONNX / TFLite conversion.

Enable it in the config:

```yaml
events:
  model: mlp
  model_path: models/event_mlp.npz
```

> **Note on accuracy.** Training on simulated bootstrap labels yields ~100 % on
> held-out simulated data, which mostly proves the features are *self-consistent*
> — it does **not** prove real-world accuracy. Re-evaluate on real, hand-checked
> clips before trusting it (`smartgym.training.benchmark` prints the confusion
> matrix so the failure mode is visible).

---

## B. Ball / shuttle detector — YOLO

### Why it exists

The classical detector relies on motion. It is fast and needs no data, but it
needs the ball to be the only small moving object — which fails with spectators,
moving shadows, or a handheld camera. A trained detector removes that assumption.

### Bootstrap labels

```bash
python -m smartgym.cli train-ball clip1.mp4,clip2.mp4 \
       --sport badminton --out data/ball
```

This writes `images/`, `labels/` (YOLO format, class 0 = ball) and `data.yaml`,
using the classical detector's high-confidence detections as seeds.

The boxes are **not** ground truth. Correct them with any labelling tool
(LabelImg, CVAT, Roboflow). Rule of thumb: 300–600 corrected frames from 3–5
different sessions beats 5 000 auto-labelled ones.

### Train

```bash
yolo detect train data=data/ball/data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

Why `yolov8n` (nano): the ball occupies a few pixels; a nano model at 640 px is
usually sufficient and runs at 30+ fps on a Jetson. If the ball is very small,
train at `imgsz=960` and enable `mosaic=0` for the last 10 epochs.

### Deploy

```yaml
detector:
  ball: yolo
  weights: models/ball_yolov8n.pt
  conf: 0.25
  imgsz: 960
  device: cuda        # or cpu
```

For the edge device, export and use TensorRT/ONNX:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```

---

## C. Evaluation protocol

Always evaluate the three layers separately:

```bash
python -m smartgym.cli benchmark --sport all --out runs/benchmark
```

```
rules        → is the rule book implemented correctly?          (target 100 %)
events_*     → are the physics heuristics right, given a track?  (physics quality)
end_to_end   → does the whole thing work on pixels?              (perception quality)
```

A drop in `rules` is a bug. A drop in `events` means the physics or thresholds
need work (or the classifier needs training). A drop in `end_to_end` means the
detector needs training. Reporting them separately is what makes progress
measurable.

### Metrics that matter

| Metric | Definition | Why |
|---|---|---|
| `winner_accuracy` | fraction of rallies where the point winner is correct | decides matches |
| `point_recall` | fraction of rallies that produced any decision | a missed rally is a missing point |
| `cause_accuracy` | fraction where the reason matches | explains the call to a human |
| `score_display_accuracy` | fraction where the whole scoreboard matches | end-to-end correctness |

`winner_accuracy` is computed over the first `min(expected, detected)` rallies,
so spurious extra points are penalised through `point_recall` instead of
invisibly inflating accuracy.
