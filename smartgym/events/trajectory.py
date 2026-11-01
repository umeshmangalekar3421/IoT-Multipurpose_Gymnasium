"""Trajectory physics: flight segments, bounces, hits and net contacts.

Why this layer exists
---------------------
A per-frame ball detector gives you dots, not semantics.  To score a match we
need the *physics* of each flight arc:

* where the ball was struck (racket contact) and by which side,
* where it touched the floor (the landing point that decides in/out),
* whether it crossed the net plane or died at the net,
* how fast it was travelling (smash speed, rally intensity).

All of it is derived from the smoothed trajectory in two spaces:

*image*  – vertical image coordinate ``y`` increases downwards, so
           *a bounce is a local maximum of y* and *an apex is a local minimum*.
           This holds for any fixed / panning-corrected camera and needs no
           depth sensor, which is what makes the method work on phone video.
*court*  – the homography gives real metres, so distances, in/out and speeds
           are physical numbers rather than pixels.

The output is a list of :class:`Segment` objects plus refined
:class:`Contact` points that the event detector turns into ``Event``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import (
    local_maxima,
    parabolic_vertex,
    savgol_smooth,
    speed,
    velocity,
)
from ..core.types import Point, TrackPoint


@dataclass
class Contact:
    """A refined physical contact between ball and floor / racket / net."""

    kind: str                  # 'bounce' | 'hit' | 'net' | 'lost'
    frame: int
    t: float
    img_xy: Point
    court_xy: Optional[Point]
    side: Optional[str] = None      # 'A' / 'B' once projected
    confidence: float = 1.0
    sub_frame_offset: float = 0.0
    speed_mps: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "frame": self.frame,
            "t": round(self.t, 4),
            "img_xy": [round(float(v), 1) for v in self.img_xy],
            "court_xy": None if self.court_xy is None else [round(float(v), 3) for v in self.court_xy],
            "side": self.side,
            "confidence": round(float(self.confidence), 3),
            "speed_mps": None if self.speed_mps is None else round(float(self.speed_mps), 2),
        }


@dataclass
class Segment:
    """One continuous flight of the ball between two contacts."""

    points: List[TrackPoint]
    t: np.ndarray                       # (N,) timestamps
    img: np.ndarray                     # (N,2) image coords
    court: Optional[np.ndarray]         # (N,2) court metres or None
    end_reason: str = "end"             # 'gap' | 'speed_jump' | 'end'
    bounces: List[Contact] = field(default_factory=list)
    crossed_net: bool = False
    min_net_distance_m: float = float("inf")
    start_side: Optional[str] = None
    end_side: Optional[str] = None
    max_speed_mps: Optional[float] = None

    @property
    def frames(self) -> List[int]:
        return [p.frame for p in self.points]

    @property
    def first_frame(self) -> int:
        return self.points[0].frame

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame

    def as_dict(self) -> dict:
        return {
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "t_start": round(float(self.t[0]), 4),
            "t_end": round(float(self.t[-1]), 4),
            "n_points": len(self.points),
            "end_reason": self.end_reason,
            "crossed_net": self.crossed_net,
            "min_net_distance_m": None if np.isinf(self.min_net_distance_m)
            else round(float(self.min_net_distance_m), 3),
            "start_side": self.start_side,
            "end_side": self.end_side,
            "max_speed_mps": None if self.max_speed_mps is None
            else round(float(self.max_speed_mps), 2),
            "bounces": [b.as_dict() for b in self.bounces],
        }


# --------------------------------------------------------------------------- #
# Segment construction
# --------------------------------------------------------------------------- #
def build_segments(track: Sequence[TrackPoint], max_gap: int = 10,
                   max_speed_mps: float = 130.0,
                   fps: float = 30.0) -> List[Segment]:
    """Split a ball track into continuous flight arcs.

    A break happens when the ball disappears for more than ``max_gap`` frames
    (occlusion, ball out of frame) or when the implied speed is unphysical
    (tracker swap / glitch).  Badminton smashes peak around 100 m/s at the
    racket and ~40 m/s in flight, tennis serves ~60 m/s, so ``max_speed_mps``
    defaults well above the plausible range.
    """
    # Interpolated points are *synthetic*: keeping them would erase the very
    # gaps that tell us the ball was struck while occluded (a racket contact).
    orig = [p for p in track if p.xy is not None]
    pts = [p for p in orig if not p.interpolated]
    if len(pts) < 2:
        pts = orig
    if len(pts) < 2:
        return []

    groups: List[List[TrackPoint]] = [[pts[0]]]
    for prev, cur in zip(pts, pts[1:]):
        gap = cur.frame - prev.frame
        dt = max(cur.t - prev.t, 1e-6)
        # speed check in court metres when available (physical), else px
        if prev.court_xy is not None and cur.court_xy is not None:
            dist = float(np.hypot(cur.court_xy[0] - prev.court_xy[0],
                                 cur.court_xy[1] - prev.court_xy[1]))
            too_fast = dist / dt > max_speed_mps
        else:
            dist = float(np.hypot(cur.xy[0] - prev.xy[0], cur.xy[1] - prev.xy[1]))
            too_fast = dist / dt > max_speed_mps * 60.0  # px proxy
        if gap > max_gap:
            groups.append([cur])
        elif too_fast:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    segments: List[Segment] = []
    for idx, group in enumerate(groups):
        if len(group) < 2:
            continue
        t = np.array([p.t for p in group], dtype=np.float64)
        img = np.array([p.xy for p in group], dtype=np.float64)
        has_court = sum(1 for p in group if p.court_xy is not None) >= max(2, len(group) // 2)
        court = (np.array([p.court_xy if p.court_xy is not None else (np.nan, np.nan)
                           for p in group], dtype=np.float64) if has_court else None)
        end_reason = "end" if idx == len(groups) - 1 else (
            "gap" if group[-1].frame >= 0 else "speed_jump")
        segments.append(Segment(points=group, t=t, img=img, court=court,
                                 end_reason=end_reason))
    return segments


# --------------------------------------------------------------------------- #
# Bounce detection
# --------------------------------------------------------------------------- #
def detect_bounces(seg: Segment, min_prominence_px: float = 8.0,
                   min_spacing_frames: int = 4,
                   smooth_window: int = 7,
                   min_speed_drop: float = 0.12,
                   min_reversal_px_s: float = 45.0) -> List[Contact]:
    """Find floor contacts as local maxima of the smoothed image-y signal.

    Two physical filters are applied on top of the peak picking so that shoe
    tips, shuttles drifting and tracker noise do not create phantom bounces:

    1. *Direction reversal* — y-velocity must flip from downward (+) to
       upward (−) around the peak.
    2. *Energy loss* — the ball loses speed at a bounce (tennis ~30-40%,
       shuttle much more). A moving object that keeps its speed is a
       *racket hit* or a panning artefact, not a bounce.
    """
    if len(seg.points) < 5:
        return []
    y = savgol_smooth(seg.img[:, 1], window=max(3, smooth_window))
    peaks = local_maxima(y, min_prominence=min_prominence_px,
                         min_distance=min_spacing_frames)
    if not peaks:
        return []

    vy = np.gradient(y, seg.t)
    # Speed is measured in IMAGE space on purpose: a ball in flight is above
    # the floor, so its floor-projected speed is meaningless.  What matters for
    # a bounce is that the object visibly loses speed at the contact.
    court_speed = speed(seg.img, seg.t)

    contacts: List[Contact] = []
    for idx in peaks:
        if idx <= 1 or idx >= len(seg.points) - 2:
            continue
        rising_before = vy[max(0, idx - 3): idx + 1].mean() > 0
        falling_after = vy[idx: min(len(vy), idx + 4)].mean() <= 0
        if not (rising_before and falling_after):
            continue

        pre = court_speed[max(0, idx - 4): idx].mean() if idx >= 2 else 0.0
        post = court_speed[idx + 1: min(len(court_speed), idx + 5)].mean()
        if pre > 1e-6 and post > 0:
            drop = 1.0 - post / pre
        else:
            drop = 1.0
        # A clear vertical reversal at speed IS the physical signature of a floor
        # contact, so it is accepted even when the ball is struck immediately
        # afterwards (which raises the post-bounce speed and hides the drop).
        strong_reversal = (abs(vy[max(0, idx - 3):idx + 1].mean()) > min_reversal_px_s
                           and abs(vy[idx:min(len(vy), idx + 4)].mean()) > min_reversal_px_s)
        is_last_peak = idx == peaks[-1]
        if not strong_reversal and drop < min_speed_drop and not is_last_peak:
            continue

        contact = refine_contact(seg, idx, kind="bounce", speed_mps=float(pre))
        contact.confidence = float(np.clip(0.55 + 0.45 * min(drop, 1.0), 0.0, 1.0))
        contact.meta["speed_drop"] = round(float(drop), 3)
        # Whether the rally ends here is decided later, from whether the ball is
        # returned (see EventDetector._mark_terminal); physics alone cannot tell
        # a tennis rally bounce from a rally-ending one.
        after = y[idx + 1: min(len(y), idx + 13)]
        rebound = float(after.max() - y[idx]) if len(after) else 0.0
        contact.meta["rebound_px"] = round(rebound, 1)
        # a ball that climbs away from the floor was still in play: either it was
        # returned, or it is a tennis ball rebounding from its first bounce
        contact.meta["rebounded"] = bool(rebound > 18.0)
        contact.meta["speed_after_px_s"] = round(float(np.mean(
            court_speed[idx + 1: min(len(court_speed), idx + 9)])), 1)
        contacts.append(contact)
    return contacts


def refine_contact(seg: Segment, idx: int, kind: str = "bounce",
                   speed_mps: Optional[float] = None) -> Contact:
    """Sub-frame refinement of a contact using a parabola through 3 samples."""
    y = seg.img[:, 1]
    lo = max(0, idx - 1)
    hi = min(len(y) - 1, idx + 1)
    off, _ = parabolic_vertex(y[lo], y[idx], y[hi])
    off = float(np.clip(off, -1.0, 1.0))

    t = float(seg.t[idx] + off * (seg.t[min(idx + 1, len(seg.t) - 1)] - seg.t[idx]))
    img = _interp_vec(seg.img, idx, off)
    court = None
    if seg.court is not None and np.all(np.isfinite(seg.court[max(0, idx - 1)])):
        court = _interp_vec(seg.court, idx, off)
        if court is not None and not np.all(np.isfinite(court)):
            court = None
    return Contact(kind=kind, frame=int(seg.points[idx].frame), t=t,
                   img_xy=(float(img[0]), float(img[1])),
                   court_xy=None if court is None else (float(court[0]), float(court[1])),
                   speed_mps=speed_mps, sub_frame_offset=off)


def _interp_vec(arr: np.ndarray, idx: int, off: float) -> np.ndarray:
    """Linear interpolation at fractional index ``idx+off``."""
    a = arr[idx]
    if off >= 0 and idx + 1 < len(arr):
        b = arr[idx + 1]
        return a + off * (b - a)
    if off < 0 and idx - 1 >= 0:
        b = arr[idx - 1]
        return a + off * (a - b)
    return a


# --------------------------------------------------------------------------- #
# Arc geometry / net analysis
# --------------------------------------------------------------------------- #
def annotate_segment(seg: Segment, net_x: float,
                     net_band_m: float = 0.35) -> Segment:
    """Fill ``crossed_net``, ``min_net_distance_m`` and side labels."""
    if seg.court is None:
        return seg
    valid = np.isfinite(seg.court).all(axis=1)
    if valid.sum() < 2:
        return seg
    xs = seg.court[valid, 0]
    d = xs - net_x
    seg.crossed_net = bool(xs.min() < net_x < xs.max())
    seg.min_net_distance_m = float(np.min(np.abs(d)))

    first = seg.court[valid][0]
    last = seg.court[valid][-1]
    seg.start_side = "A" if first[0] < net_x else "B"
    seg.end_side = "A" if last[0] < net_x else "B"

    sp = speed(seg.court, seg.t)
    sp = sp[np.isfinite(sp)]
    if len(sp):
        seg.max_speed_mps = float(np.percentile(sp, 95))
    return seg


def hit_after_net_reversal(seg: Segment, net_x: float,
                           net_band_m: float = 0.35) -> bool:
    """True when the ball approached the net and came back (net tape / post)."""
    if seg.court is None or seg.min_net_distance_m > net_band_m:
        return False
    if seg.crossed_net:
        return False
    valid = np.isfinite(seg.court).all(axis=1)
    xs = seg.court[valid, 0]
    if len(xs) < 3:
        return False
    return bool(xs.min() < xs[0] and xs[-1] < xs.max() * 1.05)


def detect_dead_ball(seg: Segment, max_speed_mps: float = 0.45,
                     min_frames: int = 8,
                     max_speed_px_s: float = 26.0) -> List[Contact]:
    """Detect the ball coming to rest — the physical end of a rally.

    A ball or shuttle that stops moving has either landed, been caught or been
    picked up; in all three cases the rally is over.  Real detectors lose the
    ball at this moment (a static object has no motion signal), and detecting
    the stop explicitly makes rally boundaries deterministic instead of relying
    on a timeout.
    """
    if len(seg.points) < min_frames + 2:
        return []
    if seg.court is not None and np.isfinite(seg.court).all(axis=1).sum() > len(seg.points) // 2:
        sp = speed(seg.court, seg.t)
        limit = max_speed_mps
    else:
        sp = speed(seg.img, seg.t)
        limit = max_speed_px_s
    sp = np.where(np.isfinite(sp), sp, np.inf)
    slow = sp < limit
    run = 0
    for i in range(len(slow)):
        run = run + 1 if slow[i] else 0
        if run >= min_frames:
            idx = i - min_frames + 1
            contact = refine_contact(seg, max(0, idx), kind="lost", speed_mps=None)
            contact.confidence = 0.75
            contact.meta["reason"] = "ball_at_rest"
            contact.meta["rest_speed"] = round(float(sp[i]), 3)
            return [contact]
    return []


def detect_hits(seg: Segment, net_x: float,
                min_speed_px_s: float = 80.0,
                smooth_window: int = 5,
                bounce_tolerance: int = 1) -> List[Contact]:
    """Detect racket contacts as reversals of the ball's *horizontal* travel.

    Discriminating a bounce from a stroke is the crux of event extraction, and
    the two are cleanly separated in image space:

    * a **bounce** reverses the *vertical* motion (down becomes up) — the ball
      stays on the same side travelling the same way horizontally;
    * a **stroke** reverses the *horizontal* motion (the ball is sent back
      towards the opponent) and usually also changes the vertical direction.

    So: bounces = y-maxima (see :func:`detect_bounces`), hits = sign changes of
    ``vx``.  Using the *image* velocity avoids the height distortion of the
    floor homography, and using pixels/second makes the threshold independent of
    camera zoom.
    """
    if len(seg.points) < 6:
        return []
    img = seg.img.astype(np.float64)
    xs = savgol_smooth(img[:, 0], window=max(3, smooth_window))
    ys = savgol_smooth(img[:, 1], window=max(3, smooth_window))
    vx = np.gradient(xs, seg.t)
    vy = np.gradient(ys, seg.t)

    y_smooth = savgol_smooth(img[:, 1], window=7)
    y_peaks = set(local_maxima(y_smooth, min_prominence=4.0, min_distance=3))

    hits: List[Contact] = []
    # the first/last few samples of a segment have unreliable velocities (the
    # ball appears/disappears), which produces phantom strokes at segment edges
    for i in range(4, len(seg.points) - 5):
        before_x = vx[max(0, i - 3):i]
        after_x = vx[i + 1:i + 4]
        before_y = vy[max(0, i - 3):i]
        after_y = vy[i + 1:i + 4]
        if len(before_x) < 2 or len(after_x) < 2:
            continue
        vb, va = float(before_x.mean()), float(after_x.mean())
        vby, vay = float(before_y.mean()), float(after_y.mean())
        near_bounce = any(abs(i - p) <= bounce_tolerance for p in y_peaks)

        reversed_x = vb * va < 0
        # a sharply turning stroke that keeps the same x-sign (a drop shot, or a
        # steep lift seen from a corner camera) is still a stroke
        nb = np.hypot(vb, vby)
        na = np.hypot(va, vay)
        turn = 0.0
        if nb > 1e-6 and na > 1e-6:
            cosang = (vb * va + vby * vay) / (nb * na)
            turn = float(np.degrees(np.arccos(float(np.clip(cosang, -1, 1)))))
        reversed_y = vby * vay < 0

        if not reversed_x:
            # without an x-reversal, only a very sharp turn *away from a floor
            # contact* counts; a y-peak explains the change as a bounce
            if near_bounce or turn < 100.0 or not reversed_y:
                continue
        elif near_bounce:
            continue
        # a stroke is characterised by the *fast* side: right after a contact the
        # ball is momentarily slow, so requiring both sides to be fast would miss
        # every return struck from a low position
        if max(nb, na) < min_speed_px_s:
            continue
        contact = refine_contact(seg, i, kind="hit", speed_mps=None)
        sprite = float(np.hypot(vx[i], vy[i]))
        contact.meta["vx_before_px_s"] = round(vb, 1)
        contact.meta["vx_after_px_s"] = round(va, 1)
        contact.meta["turn_deg"] = round(turn, 1)
        contact.meta["speed_px_s"] = round(sprite, 1)
        if seg.court is not None and np.isfinite(seg.court[i]).all():
            contact.court_xy = (float(seg.court[i, 0]), float(seg.court[i, 1]))
            contact.side = "A" if seg.court[i, 0] < net_x else "B"
        contact.confidence = float(np.clip(
            0.5 + 0.3 * min(1.0, min(abs(vb), abs(va)) / 500.0), 0.3, 0.95))
        hits.append(contact)
    return hits


def detect_hits_image_space(seg: Segment, min_speed_px: float = 25.0) -> List[Contact]:
    """Fallback hit detector when the court is not calibrated (image space)."""
    if len(seg.points) < 5:
        return []
    x = savgol_smooth(seg.img[:, 0], window=5)
    vx = np.gradient(x, seg.t)
    hits: List[Contact] = []
    for i in range(2, len(vx) - 2):
        before, after = vx[i - 2:i].mean(), vx[i + 1:i + 3].mean()
        if before * after >= 0:
            continue
        if min(abs(before), abs(after)) < min_speed_px:
            continue
        c = refine_contact(seg, i, kind="hit", speed_mps=None)
        c.confidence = 0.45
        c.meta["space"] = "image"
        hits.append(c)
    return hits


def _opening_hit(seg: Segment) -> Contact:
    """The ball appeared in mid-air: somebody must have struck it."""
    first = seg.points[0]
    court_xy = None
    if seg.court is not None and np.isfinite(seg.court[0]).all():
        court_xy = (float(seg.court[0, 0]), float(seg.court[0, 1]))
    return Contact(
        kind="hit",
        frame=first.frame,
        t=float(first.t),
        img_xy=(float(seg.img[0, 0]), float(seg.img[0, 1])),
        court_xy=court_xy,
        side=seg.start_side,
        confidence=0.5,
        meta={"reason": "struck_before_track_start"},
    )


def segment_contacts(segments: Sequence[Segment], net_x: float,
                     bounce_kwargs: Optional[dict] = None,
                     net_band_m: float = 0.35,
                     detect_hit_events: bool = True) -> List[Contact]:
    """All physical contacts, in chronological order, across the whole track.

    * bounces — local maxima of the smoothed image-y signal inside a run,
    * hits — reversals of the horizontal court velocity inside a run,
    * re-acquisition hits — the ball reappears shortly after being lost, which
      means it was struck while occluded (racket contact),
    * net contacts — an arc that dies at the net plane,
    * lost — the ball left the tracking volume (rally ends / out of play).
    """
    bounce_kwargs = bounce_kwargs or {}
    contacts: List[Contact] = []
    prev: Optional[Segment] = None

    for seg in segments:
        if prev is not None:
            gap = seg.first_frame - prev.last_frame
            if 1 <= gap <= 12:
                last = prev.points[-1]
                contacts.append(Contact(
                    kind="hit",
                    frame=seg.first_frame,
                    t=float(seg.t[0]),
                    img_xy=(float(seg.img[0, 0]), float(seg.img[0, 1])),
                    court_xy=(None if seg.court is None or not np.isfinite(seg.court[0]).all()
                              else (float(seg.court[0, 0]), float(seg.court[0, 1]))),
                    side=seg.start_side,
                    confidence=0.7,
                    meta={"gap_frames": int(gap), "from_frame": int(last.frame),
                          "reason": "reacquired_after_gap"},
                ))
            elif gap > 12:
                contacts.append(_lost_contact(prev))
                # the ball reappears far away in time: someone struck it while
                # it was invisible (out of frame / occluded).  Emitting a hit
                # here is what tells the rule engine the rally continued.
                contacts.append(Contact(
                    kind="hit",
                    frame=seg.first_frame,
                    t=float(seg.t[0]),
                    img_xy=(float(seg.img[0, 0]), float(seg.img[0, 1])),
                    court_xy=(None if seg.court is None or not np.isfinite(seg.court[0]).all()
                              else (float(seg.court[0, 0]), float(seg.court[0, 1]))),
                    side=seg.start_side,
                    confidence=0.45,
                    meta={"gap_frames": int(gap), "reason": "struck_while_invisible"},
                ))

        if prev is None and seg.first_frame > 2:
            # the very first flight of the clip: the ball appeared in mid-air
            contacts.append(_opening_hit(seg))

        seg.bounces = detect_bounces(seg, **bounce_kwargs)
        contacts.extend(seg.bounces)
        dead = detect_dead_ball(seg)
        contacts.extend(dead)
        if detect_hit_events:
            hits = detect_hits(seg, net_x)
            contacts.extend(hits)
        prev = seg

    if prev is not None:
        contacts.append(_lost_contact(prev))

    for seg in segments[:-1]:
        if hit_after_net_reversal(seg, net_x, net_band_m):
            last = seg.points[-1]
            contacts.append(Contact(
                kind="net",
                frame=last.frame,
                t=float(seg.t[-1]),
                img_xy=(float(seg.img[-1, 0]), float(seg.img[-1, 1])),
                court_xy=(None if seg.court is None or not np.isfinite(seg.court[-1]).all()
                          else (float(seg.court[-1, 0]), float(seg.court[-1, 1]))),
                side=seg.end_side,
                confidence=0.6,
                meta={"min_net_distance_m": round(float(seg.min_net_distance_m), 3)},
            ))
    contacts.sort(key=lambda c: (c.t, c.frame))
    return contacts


def extrapolate_landing(seg: Segment, horizon_s: float = 2.0,
                        min_points: int = 6) -> Optional[Tuple[float, float]]:
    """Predict where a ball that vanished mid-flight will touch the floor.

    When the ball leaves the frame (or the detector loses it) it is usually
    still airborne, so its *current* floor projection is far from the real
    landing point — and the in/out call depends on that point to within
    centimetres.  Fitting a parabola to the recent vertical motion and taking
    its vertex gives the floor contact under constant-gravity motion; the
    horizontal position is interpolated linearly to the same instant.
    """
    n = len(seg.points)
    if n < min_points:
        return None
    k = min(n, 15)
    t = seg.t[-k:].astype(np.float64)
    y = savgol_smooth(seg.img[-k:, 1], window=min(7, k if k % 2 else k - 1))
    x = seg.img[-k:, 0]
    t0 = t - t[-1]
    try:
        cy = np.polyfit(t0, y, 2)
        cx = np.polyfit(t0, x, 1)
    except Exception:
        return None
    if cy[0] <= 1e-6:
        return None                       # not a downward parabola
    dt_vertex = -cy[1] / (2 * cy[0])
    if not (0.0 < dt_vertex <= horizon_s):
        return None
    # a ball that is still climbing will not land within the horizon
    vy_now = cy[1]
    if vy_now < 0:
        return None
    x_land = float(np.polyval(cx, dt_vertex))
    y_land = float(np.polyval(cy, dt_vertex))
    return x_land, y_land


def _lost_contact(seg: Segment) -> Contact:
    last = seg.points[-1]
    court = None
    if seg.court is not None and np.isfinite(seg.court[-1]).all():
        court = (float(seg.court[-1, 0]), float(seg.court[-1, 1]))
    return Contact(
        kind="lost",
        frame=last.frame,
        t=float(seg.t[-1]),
        img_xy=(float(seg.img[-1, 0]), float(seg.img[-1, 1])),
        court_xy=court,
        side=seg.end_side,
        confidence=0.5,
        meta={"end_reason": seg.end_reason,
              "crossed_net": bool(seg.crossed_net)},
    )
