"""Offline track-before-detect: link per-frame candidates into ball tracks.

The online Kalman tracker is the right tool for a *live* stream, where you must
decide at each frame.  For recorded video — which is what scoring and evidence
need — you can do much better by looking at all candidates at once:

1. **Link** candidates frame-to-frame with a velocity-aware cost (a ball keeps
   flying in a straight-ish line at a roughly constant speed, so its next
   position is predictable; a flickering line or a spectator is not).
2. **Score** every resulting track by length, confidence and how far it
   actually *travelled* (a track that never leaves one spot is scenery).
3. **Select** the tracks that look like a ball in play, discarding overlapping
   decoys.

This is deliberately not a deep model: it is a transparent, tunable,
CPU-real-time algorithm that fails in ways you can explain and fix - which is
exactly what a course project needs.  All thresholds are physical (pixels per
second) and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Detection, Point, TrackPoint


@dataclass
class Candidate:
    frame: int
    t: float
    xy: Point
    conf: float = 1.0
    box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Track:
    """One linked ball track."""

    points: List[Candidate] = field(default_factory=list)
    score: float = 0.0

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def first_frame(self) -> int:
        return self.points[0].frame

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame

    def displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(self.points, self.points[1:]):
            total += float(np.hypot(b.xy[0] - a.xy[0], b.xy[1] - a.xy[1]))
        return total

    def mean_speed_px_s(self) -> float:
        if len(self.points) < 2:
            return 0.0
        dt = max(1e-6, self.points[-1].t - self.points[0].t)
        return self.displacement() / dt

    def velocity_at_end(self, fps: float) -> Tuple[float, float]:
        """Robust velocity estimate (px/s) from the last few points."""
        pts = self.points[-3:]
        if len(pts) < 2:
            return 0.0, 0.0
        t0, t1 = pts[0].t, pts[-1].t
        dt = max(1e-6, t1 - t0)
        return ((pts[-1].xy[0] - pts[0].xy[0]) / dt,
                (pts[-1].xy[1] - pts[0].xy[1]) / dt)


class OfflineLinker:
    """Velocity-aware, greedy global linker with plausibility-based selection."""

    def __init__(self, fps: float = 30.0, max_gap: int = 10,
                 max_speed_px_s: float = 3200.0,
                 gate_px: float = 140.0,
                 min_track_points: int = 6,
                 min_displacement_px: float = 90.0,
                 min_mean_speed_px_s: float = 110.0):
        self.fps = float(fps)
        self.max_gap = int(max_gap)
        self.max_speed = float(max_speed_px_s)
        self.gate = float(gate_px)
        self.min_track_points = int(min_track_points)
        self.min_displacement = float(min_displacement_px)
        self.min_mean_speed = float(min_mean_speed_px_s)

    # -- linking ------------------------------------------------------------
    def link(self, candidates: Dict[int, Sequence[Candidate]]) -> List[Track]:
        active: List[Track] = []
        finished: List[Track] = []
        frames = sorted(candidates)

        for frame in frames:
            t = None
            cands = list(candidates[frame])
            if cands:
                t = cands[0].t

            # close stale tracks
            still: List[Track] = []
            for tr in active:
                if frame - tr.last_frame > self.max_gap:
                    finished.append(tr)
                else:
                    still.append(tr)
            active = still

            if not cands:
                continue

            # cost matrix (track, candidate)
            pairs: List[Tuple[float, int, int]] = []
            for ti, tr in enumerate(active):
                vx, vy = tr.velocity_at_end(self.fps)
                speed = float(np.hypot(vx, vy))
                if speed > self.max_speed:
                    continue
                last = tr.points[-1]
                dt = max(1e-3, t - last.t)
                pred = (last.xy[0] + vx * dt, last.xy[1] + vy * dt)
                for ci, c in enumerate(cands):
                    dist = float(np.hypot(c.xy[0] - pred[0], c.xy[1] - pred[1]))
                    # implied speed must stay physical
                    implied = float(np.hypot(c.xy[0] - last.xy[0],
                                             c.xy[1] - last.xy[1])) / dt
                    if implied > self.max_speed:
                        continue
                    if dist > self.gate:
                        continue
                    # constant-velocity consistency is the main signal; a
                    # confident candidate is preferred on ties
                    cost = dist - 45.0 * float(c.conf)
                    pairs.append((cost, ti, ci))

            pairs.sort(key=lambda p: p[0])
            used_t: set = set()
            used_c: set = set()
            for cost, ti, ci in pairs:
                if ti in used_t or ci in used_c:
                    continue
                used_t.add(ti)
                used_c.add(ci)
                active[ti].points.append(cands[ci])

            for ci, c in enumerate(cands):
                if ci in used_c:
                    continue
                active.append(Track(points=[c]))

        finished.extend(active)
        return finished

    # -- scoring & selection ------------------------------------------------
    def score(self, tr: Track) -> float:
        if tr.length < 2:
            return 0.0
        span = max(1e-6, tr.points[-1].t - tr.points[0].t)
        mean_conf = float(np.mean([c.conf for c in tr.points]))
        dist = tr.displacement()
        # reward long, confident, fast tracks; penalise parked ones
        return (tr.length ** 0.5) * mean_conf * (1.0 + dist / 400.0) * (1.0 + span)

    def plausible(self, tr: Track) -> bool:
        return (tr.length >= self.min_track_points
                and tr.displacement() >= self.min_displacement
                and tr.mean_speed_px_s() >= self.min_mean_speed)

    def select(self, tracks: Sequence[Track]) -> List[Track]:
        """Keep plausible tracks, dropping those overlapping a better one."""
        good = [t for t in tracks if self.plausible(t)]
        for t in good:
            t.score = self.score(t)
        good.sort(key=lambda t: -t.score)
        chosen: List[Track] = []
        for tr in good:
            if any(not (tr.last_frame < c.first_frame or tr.first_frame > c.last_frame)
                   for c in chosen):
                continue
            chosen.append(tr)
        chosen.sort(key=lambda t: t.first_frame)
        return chosen

    # -- convenience --------------------------------------------------------
    def run(self, candidates: Dict[int, Sequence[Candidate]]) -> List[Track]:
        return self.select(self.link(candidates))


def to_track_points(tracks: Sequence[Track], court=None) -> List[TrackPoint]:
    """Flatten selected tracks into the pipeline's ``TrackPoint`` list."""
    out: List[TrackPoint] = []
    for tr in tracks:
        for c in tr.points:
            tp = TrackPoint(frame=c.frame, t=c.t, xy=c.xy, conf=c.conf)
            if court is not None and getattr(court, "registered", False):
                tp.court_xy = court.to_court_point(c.xy)
            out.append(tp)
    out.sort(key=lambda tp: tp.frame)
    return out
