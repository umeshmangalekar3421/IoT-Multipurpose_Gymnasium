"""A small multi-layer perceptron for contact classification — NumPy only.

Why not PyTorch?  Because the deliverable has to *run on the edge device*.  A
16-24-3 MLP is a few hundred parameters; implemented in NumPy it trains in a
second on a laptop and infers in microseconds with no framework, no CUDA and no
200 MB wheel.  It can later be exported to TensorFlow Lite / ONNX with
:meth:`EventClassifier.export_onnx_weights` if a bigger model is needed.

The classifier answers one question: *is this detected contact a floor bounce, a
racket hit, or spurious noise?*  It is blended with the physics confidence by
:class:`smartgym.events.detector.EventDetector`, never used on its own — a
poorly calibrated model then degrades gracefully.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES, LABELS, N_FEATURES


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class EventClassifier:
    """Two-hidden-layer MLP with standardised inputs, trained by Adam."""

    n_features: int = N_FEATURES
    hidden: int = 24
    n_classes: int = len(LABELS)
    weights: Optional[List[np.ndarray]] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    labels: Tuple[str, ...] = LABELS

    # -- inference ----------------------------------------------------------
    def _init_if_needed(self) -> None:
        if self.weights is not None:
            return
        rng = np.random.default_rng(0)
        h = self.hidden
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / self.n_features), (self.n_features, h)),
            np.zeros(h),
            rng.normal(0, np.sqrt(2.0 / h), (h, h // 2 or 1)),
            np.zeros(h // 2 or 1),
            rng.normal(0, np.sqrt(2.0 / (h // 2 or 1)), (h // 2 or 1, self.n_classes)),
            np.zeros(self.n_classes),
        ]
        self.mean = np.zeros(self.n_features)
        self.std = np.ones(self.n_features)

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, self.n_features)
        mean = self.mean if self.mean is not None else np.zeros(self.n_features)
        std = self.std if self.std is not None else np.ones(self.n_features)
        return (X - mean) / np.where(std < 1e-6, 1.0, std)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self._init_if_needed()
        W1, b1, W2, b2, W3, b3 = self.weights
        z1 = self._standardise(X) @ W1 + b1
        a1 = _relu(z1)
        z2 = a1 @ W2 + b2
        a2 = _relu(z2)
        z3 = a2 @ W3 + b3
        return _softmax(z3), {"a1": a1, "a2": a2, "z1": z1, "z2": z2}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class (``bounce``) for each row."""
        probs, _ = self._forward(X)
        return probs[:, 0]

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_all(X), axis=1)

    # -- training -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: Sequence[int], epochs: int = 300,
            lr: float = 0.02, batch_size: int = 32, seed: int = 0,
            class_weight: Optional[Dict[int, float]] = None,
            val_split: float = 0.15) -> Dict[str, Any]:
        """Train with Adam + cross-entropy; returns metrics."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, self.n_features)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(X) < 8:
            raise ValueError(f"need at least 8 samples, got {len(X)}")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]
        n_val = int(len(X) * val_split)
        X_val, y_val = X[:n_val], y[:n_val]
        X_tr, y_tr = X[n_val:], y[n_val:]
        if len(X_tr) < 4:
            X_tr, y_tr, X_val, y_val = X, y, X[:min(4, len(X))], y[:min(4, len(y))]

        self.mean = X_tr.mean(axis=0)
        self.std = X_tr.std(axis=0) + 1e-6
        self._init_if_needed()

        # class weights counter unbalanced contact types (bounces are rarer)
        counts = np.bincount(y_tr, minlength=self.n_classes).astype(np.float64)
        cw = np.ones(self.n_classes)
        for c in range(self.n_classes):
            if counts[c] > 0:
                cw[c] = len(y_tr) / (self.n_classes * counts[c])
        if class_weight:
            for k, v in class_weight.items():
                cw[int(k)] = float(v)

        params = self.weights
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), batch_size):
                batch = idx[start:start + batch_size]
                xb, yb = X_tr[batch], y_tr[batch]
                probs, cache = self._forward(xb)
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0
                w = cw[yb][:, None]
                d3 = (probs - onehot) * w / len(xb)
                W1, bb1, W2, bb2, W3, bb3 = params
                gW3 = cache["a2"].T @ d3
                gb3 = d3.sum(axis=0)
                d2 = (d3 @ W3.T) * (cache["z2"] > 0)
                gW2 = cache["a1"].T @ d2
                gb2 = d2.sum(axis=0)
                d1 = (d2 @ W2.T) * (cache["z1"] > 0)
                gW1 = self._standardise(xb).T @ d1
                gb1 = d1.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3]
                for i, g in enumerate(grads):
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * (g * g)
                    mhat = m[i] / (1 - b1 ** epoch)
                    vhat = v[i] / (1 - b2 ** epoch)
                    params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            if epoch % 25 == 0 or epoch == epochs:
                train_loss = self._loss(X_tr, y_tr, cw)
                history.append({"epoch": epoch, "loss": round(float(train_loss), 4)})

        metrics = {
            "epochs": epochs,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "train_loss": round(float(self._loss(X_tr, y_tr, cw)), 4),
            "train_accuracy": round(float((self.predict(X_tr) == y_tr).mean()), 4),
            "features": list(FEATURE_NAMES),
            "labels": list(LABELS),
            "confusion": self._confusion(X_val, y_val),
        }
        if len(X_val) >= 4:
            metrics["val_accuracy"] = round(float((self.predict(X_val) == y_val).mean()), 4)
        metrics["history"] = history
        return metrics

    def _loss(self, X: np.ndarray, y: np.ndarray, cw: Optional[np.ndarray] = None) -> float:
        probs, _ = self._forward(X)
        p = np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)
        w = np.ones(len(y)) if cw is None else cw[y]
        return float(-(w * np.log(p)).sum() / max(1e-9, w.sum()))

    def _confusion(self, X: np.ndarray, y: np.ndarray) -> List[List[int]]:
        mat = np.zeros((self.n_classes, self.n_classes), dtype=int)
        pred = self.predict(X)
        for t, p in zip(y, pred):
            mat[int(t), int(p)] += 1
        return mat.tolist()

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {f"w{i}": w for i, w in enumerate(self.weights or [])}
        payload.update({
            "mean": self.mean, "std": self.std,
            "meta": np.array([self.n_features, self.hidden, self.n_classes]),
            "labels": np.array(self.labels),
        })
        if str(path).endswith(".npz"):
            np.savez(path, **payload)
        else:  # human-readable JSON fallback
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in payload.items()}, fh)
        return path

    @classmethod
    def load(cls, path: str) -> "EventClassifier":
        if str(path).endswith(".npz"):
            data = np.load(path, allow_pickle=False)
            meta = data["meta"]
            clf = cls(n_features=int(meta[0]), hidden=int(meta[1]),
                      n_classes=int(meta[2]))
            clf.weights = [data[f"w{i}"] for i in range(6)]
            clf.mean = data["mean"]
            clf.std = data["std"]
            clf.labels = tuple(str(x) for x in data["labels"])
            return clf
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload["meta"]
        clf = cls(n_features=int(meta[0]), hidden=int(meta[1]), n_classes=int(meta[2]))
        clf.weights = [np.asarray(payload[f"w{i}"]) for i in range(6)]
        clf.mean = np.asarray(payload["mean"])
        clf.std = np.asarray(payload["std"])
        clf.labels = tuple(payload["labels"])
        return clf

    def export_onnx_weights(self, path: str) -> str:
        """Dump the weights in a framework-neutral form for ONNX/TFLite export."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "input_dim": self.n_features,
            "hidden": self.hidden,
            "classes": list(self.labels),
            "input_mean": (self.mean or np.zeros(self.n_features)).tolist(),
            "input_std": (self.std or np.ones(self.n_features)).tolist(),
            "layers": [
                {"type": "dense", "W": w.tolist(), "b": b.tolist(),
                 "activation": ("relu" if i < 4 else "softmax")}
                for i, (w, b) in enumerate(zip((self.weights or [])[0::2],
                                               (self.weights or [])[1::2]))
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a feature CSV produced by ``smartgym.cli export-dataset``."""
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, labels = [], []
        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(int(row["label"]))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def train_from_csv(csv_path: str, out_path: str = "models/event_mlp.npz",
                   epochs: int = 300, hidden: int = 24, seed: int = 0) -> Dict[str, Any]:
    X, y = load_csv(csv_path)
    clf = EventClassifier(hidden=hidden)
    metrics = clf.fit(X, y, epochs=epochs, seed=seed)
    clf.save(out_path)
    metrics["model_path"] = out_path
    metrics["n_samples"] = int(len(X))
    return metrics
