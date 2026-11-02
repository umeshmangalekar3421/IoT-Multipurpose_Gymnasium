"""Constant-velocity Kalman filter + tracklet assembly for the ball.

The ball is tiny, motion-blurred and frequently occluded, so a single-frame
detector is never enough.  The tracker does three jobs:

1. **Gating** – reject detections that jump implausibly far from the prediction.
2. **Coasting** – keep a track alive through short occlusions (players, net).
3. **Interpolation** – fill gaps so the trajectory smoother sees a continuous
   signal (bounces are then clean local maxima).

The maths is deliberately simple (per-axis constant velocity) so it runs at
hundreds of FPS on a Jetson Nano class device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import Detection, TrackPoint


class BallKalman:
    """4-state (x, y, vx, vy) Kalman filter with a constant-velocity model."""

    def __init__(self, pos: Sequence[float], dt: float = 1 / 30,
                 process_noise: float = 1.5, meas_noise: float = 6.0):
        self.dt = dt
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 100.0
        q = process_noise
        self.Q = np.diag([q * dt, q * dt, q * 4, q * 4]) ** 2 * 10.0
        self.R = np.eye(2) * (meas_noise ** 2)
        self.F = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: Sequence[float]) -> None:
        z = np.asarray(z, dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def innovation(self, z: Sequence[float]) -> float:
        """Mahalanobis-ish gating distance in pixels."""
        S = self.H @ self.P @ self.H.T + self.R
        d = np.asarray(z, dtype=np.float64) - self.H @ self.x
        try:
            return float(np.sqrt(d @ np.linalg.solve(S, d)))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(d))


@dataclass
class BallTracker:
    """Greedy nearest-neighbour association with Kalman gating + coasting."""

    dt: float = 1 / 30
    gate_px: float = 90.0          # max association distance (Mahalanobis)
    max_coast_frames: int = 12     # how long a lost track survives
    reacquire_px: float = 220.0    # relaxed gate when re-acquiring after a gap
    # A ball in play is never stationary.  If a track stops moving for this
    # many frames it is a distractor (a player's head, a shoe, a reflection)
    # or a dead ball, and the track is dropped so the search re-opens.
    min_speed_px_s: float = 45.0
    max_static_frames: int = 15
    stasis_window: int = 10        # frames used to measure net displacement
    stasis_px: float = 35.0        # net displacement below this = stationary
    _kf: Optional[BallKalman] = None
    _lost: int = 0
    _age: int = 0
    _slow: int = 0
    _hist_xy: List[Tuple[float, float]] = field(default_factory=list)
    _history: List[TrackPoint] = field(default_factory=list)

    # -- public API ---------------------------------------------------------
    def step(self, frame: int, t: float,
             detections: Sequence[Detection]) -> Optional[TrackPoint]:
        """Feed one frame; return the ball observation (real or coasted)."""
        cand = [d for d in detections if d.label == "ball"] or list(detections)

        if self._kf is None:
            if not cand:
                return None
            best = max(cand, key=lambda d: d.conf)
            self._kf = BallKalman(best.xy, self.dt, meas_noise=max(2.0, 12.0 * (1 - best.conf) + 2))
            self._age = 0
            tp = TrackPoint(frame=frame, t=t, xy=best.xy, conf=best.conf)
            self._history.append(tp)
            return tp

        self._kf.predict()
        pred = self._kf.pos
        self._age += 1

        # Speed-aware gating.  A fast ball is *predictable*, so a tight window
        # around the prediction is both safer (it rejects stationary
        # distractors such as a player's head) and sufficient.  Right after
        # (re)acquisition the velocity estimate is still poor, so fall back to
        # the generous gate.
        pred_speed = float(np.hypot(*self._kf.vel))
        if self._lost > 0:
            gate = self.reacquire_px
        elif self._age <= 3:
            gate = self.gate_px
        else:
            gate = float(np.clip(0.35 * pred_speed * self.dt + 22.0,
                                 26.0, self.gate_px))

        scored: List[Tuple[float, Detection]] = []
        for d in cand:
            dist = float(np.hypot(d.xy[0] - pred[0], d.xy[1] - pred[1]))
            inflate = 1.0 + 2.0 * (1.0 - float(d.conf))  # tolerate low-confidence jumps less
            if dist <= gate * inflate:
                scored.append((dist - 30.0 * float(d.conf), d))

        if scored:
            scored.sort(key=lambda s: s[0])
            _, best = scored[0]
            self._kf.update(best.xy)
            self._lost = 0
            tp = TrackPoint(frame=frame, t=t, xy=best.xy, conf=best.conf)
        else:
            self._lost += 1
            if self._lost > self.max_coast_frames:
                self.reset()
                return None
            tp = TrackPoint(frame=frame, t=t, xy=pred, conf=0.0, interpolated=True)

        # --- distractor rejection ---------------------------------------
        # A real ball keeps moving in the same direction; a distractor (a
        # player's head, a shoe, a reflection) jitters around a fixed point.
        # Comparing the position now with ~1/3 s ago distinguishes the two far
        # better than the instantaneous Kalman velocity, which is fooled by
        # jitter between two nearby blobs.
        if self.max_static_frames > 0 and not tp.interpolated:
            self._hist_xy.append(tp.xy)
            if len(self._hist_xy) > self.stasis_window + 1:
                self._hist_xy.pop(0)
            if len(self._hist_xy) == self.stasis_window + 1:
                dx = tp.xy[0] - self._hist_xy[0][0]
                dy = tp.xy[1] - self._hist_xy[0][1]
                net = float(np.hypot(dx, dy))
                path = 0.0
                for a, b in zip(self._hist_xy, self._hist_xy[1:]):
                    path += float(np.hypot(b[0] - a[0], b[1] - a[1]))
                # a jittering blob travels a long path but does not get far
                jittering = path > 1e-6 and net < 0.25 * path
                if net < self.stasis_px or jittering:
                    self._slow += 1
                else:
                    self._slow = 0
                if self._slow > self.max_static_frames:
                    self.reset()
                    return None

        self._history.append(tp)
        return tp

    def reset(self) -> None:
        self._kf = None
        self._lost = 0
        self._age = 0
        self._slow = 0
        self._hist_xy = []

    @property
    def history(self) -> List[TrackPoint]:
        return self._history


def interpolate_gaps(track: List[TrackPoint], max_gap: int = 12) -> List[TrackPoint]:
    """Linearly fill missing frames inside the track's frame span."""
    if len(track) < 2:
        return track
    by_frame = {tp.frame: tp for tp in track}
    frames = sorted(by_frame)
    filled: List[TrackPoint] = []
    for a, b in zip(frames, frames[1:]):
        filled.append(by_frame[a])
        gap = b - a
        if 1 < gap <= max_gap + 1:
            pa, pb = np.array(by_frame[a].xy), np.array(by_frame[b].xy)
            for k in range(1, gap):
                alpha = k / gap
                xy = pa + alpha * (pb - pa)
                t = by_frame[a].t + alpha * (by_frame[b].t - by_frame[a].t)
                filled.append(TrackPoint(frame=a + k, t=t,
                                         xy=(float(xy[0]), float(xy[1])),
                                         court_xy=None, conf=0.0, interpolated=True))
    filled.append(by_frame[frames[-1]])
    return filled


def split_tracklets(track: List[TrackPoint], max_gap: int = 10,
                    max_speed_px_s: float = 6000.0) -> List[List[TrackPoint]]:
    """Split a track into physically plausible flight segments.

    A segment breaks when the ball is missing for ``max_gap`` frames or when the
    implied speed is unphysical (a tracking glitch / swap between two balls).
    """
    if not track:
        return []
    segments: List[List[TrackPoint]] = [[track[0]]]
    for prev, cur in zip(track, track[1:]):
        gap = cur.frame - prev.frame
        dt = max(cur.t - prev.t, 1e-6)
        spd = float(np.hypot(cur.xy[0] - prev.xy[0], cur.xy[1] - prev.xy[1]) / dt)
        if gap > max_gap or spd > max_speed_px_s:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return [s for s in segments if len(s) >= 2]
