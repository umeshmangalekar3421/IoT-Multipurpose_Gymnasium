"""Feature engineering for the learned event classifier.

The hand-written event detector (``smartgym.events.trajectory``) decides
bounce-vs-hit from a handful of thresholds.  Those thresholds are the weakest
part of the system: they were tuned for one camera and one pace of play.

The classifier here learns the same decision from *features* computed around a
candidate contact, so the thresholds are replaced by a model that can be
retrained per venue.  Features are deliberately physical and scale-free
(pixels/second normalised by the apparent ball size, angles in degrees, court
metres) so a model trained on one camera generalises reasonably to another.

Feature vector (16 values)
--------------------------
=====  =====================================================================
 0      speed before contact (px/s, normalised by image diagonal)
 1      speed after contact
 2      speed ratio after/before
 3      vertical velocity before (normalised)
 4      vertical velocity after (normalised)
 5      horizontal velocity before (signed, normalised)
 6      horizontal velocity after (signed, normalised)
 7      turn angle between incoming and outgoing velocity (deg / 180)
 8      local maximum of the y signal at the contact (bounce signature)
 9      minimum of the y signal nearby (apex signature)
10      distance to the nearest painted line, court metres
11      distance to the net, court metres
12      height proxy: image y relative to the court's baseline range
13      contact confidence reported by the detector
14      frames since the previous contact (normalised, capped at 60)
15      ball is on the far half (0) or near half (1)
=====  =====================================================================
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import local_maxima, savgol_smooth, speed
from ..court.model import CourtModel

FEATURE_NAMES: Tuple[str, ...] = (
    "speed_before_n", "speed_after_n", "speed_ratio", "vy_before_n", "vy_after_n",
    "vx_before_n", "vx_after_n", "turn_angle", "is_local_y_max", "is_local_y_min",
    "dist_to_line_m", "dist_to_net_m", "y_rel", "detector_conf",
    "frames_since_prev", "near_half",
)

N_FEATURES = len(FEATURE_NAMES)


def contact_features(seg: Any, contact: Any, court: CourtModel,
                     fps: float = 30.0, prev_contact_frame: Optional[int] = None
                     ) -> np.ndarray:
    """Build the feature vector for one candidate contact.

    ``seg`` is a :class:`smartgym.events.trajectory.Segment`, ``contact`` a
    :class:`~smartgym.events.trajectory.Contact`.
    """
    img = np.asarray(seg.img, dtype=np.float64)
    t = np.asarray(seg.t, dtype=np.float64)
    if len(img) < 5:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # locate the contact index inside the segment
    idx = 0
    best = float("inf")
    for i, p in enumerate(seg.points):
        d = abs(p.frame - contact.frame)
        if d < best:
            best, idx = d, i

    diag = float(np.hypot(*court_scale(court, seg))) or 1.0
    xs = savgol_smooth(img[:, 0], 5)
    ys = savgol_smooth(img[:, 1], 5)
    vx = np.gradient(xs, t)
    vy = np.gradient(ys, t)
    sp = np.hypot(vx, vy)

    before = slice(max(0, idx - 4), max(1, idx))
    after = slice(min(len(sp) - 1, idx + 1), min(len(sp), idx + 5))
    vbx = float(np.mean(vx[before])) if idx > 0 else 0.0
    vby = float(np.mean(vy[before])) if idx > 0 else 0.0
    vax = float(np.mean(vx[after])) if idx < len(sp) - 1 else 0.0
    vay = float(np.mean(vy[after])) if idx < len(sp) - 1 else 0.0
    sb = float(np.hypot(vbx, vby)) / diag
    sa = float(np.hypot(vax, vay)) / diag
    ratio = sa / sb if sb > 1e-9 else 0.0

    denom = (sb * sa) or 1e-9
    cosang = float(np.clip((vbx * vax + vby * vay) / (denom * diag * diag), -1, 1))
    angle = float(np.degrees(np.arccos(cosang))) / 180.0 if denom > 1e-9 else 0.5

    window = ys[max(0, idx - 4): min(len(ys), idx + 5)]
    is_max = 1.0 if len(window) and ys[idx] >= window.max() - 1e-6 else 0.0
    is_min = 1.0 if len(window) and ys[idx] <= window.min() + 1e-6 else 0.0

    court_xy = contact.court_xy
    if court_xy is None and court.registered:
        court_xy = court.to_court_point(contact.img_xy)
    dist_line = (court.distance_to_nearest_line(court_xy, "singles")
                 if court_xy is not None and court.registered else 2.0)
    dist_net = (abs(court_xy[0] - court.spec.net_x)
                if court_xy is not None else 5.0)

    y_min, y_max = float(ys.min()), float(ys.max())
    y_rel = (float(ys[idx]) - y_min) / max(1e-6, y_max - y_min)

    gap = 60
    if prev_contact_frame is not None:
        gap = int(min(60, max(0, contact.frame - prev_contact_frame)))

    near_half = 1.0
    if court_xy is not None and court.registered:
        near_half = 1.0 if court_xy[0] > court.spec.net_x else 0.0

    return np.array([
        sb, sa, float(np.clip(ratio, 0, 5)), vby / diag, vay / diag,
        vbx / diag, vax / diag, angle, is_max, is_min,
        float(min(dist_line, 5.0)), float(min(dist_net, 15.0)),
        float(np.clip(y_rel, 0, 1)), float(contact.confidence),
        gap / 60.0, near_half,
    ], dtype=np.float32)


def court_scale(court: CourtModel, seg: Any) -> Tuple[int, int]:
    """Image diagonal proxy: spread of the segment points plus the frame size."""
    img = np.asarray(seg.img, dtype=np.float64)
    w = float(img[:, 0].max() - img[:, 0].min()) if len(img) else 1280.0
    h = float(img[:, 1].max() - img[:, 1].min()) if len(img) else 720.0
    return int(max(w, 640)), int(max(h, 360))


def features_for_detection(segments: Sequence[Any], contacts: Sequence[Any],
                           court: CourtModel, fps: float = 30.0) -> np.ndarray:
    """Feature matrix for a whole detection (rows aligned with ``contacts``)."""
    rows: List[np.ndarray] = []
    prev_frame = None
    for c in sorted(contacts, key=lambda c: c.frame):
        seg = next((s for s in segments if s.first_frame <= c.frame <= s.last_frame), None)
        if seg is None:
            rows.append(np.zeros(N_FEATURES, dtype=np.float32))
            continue
        rows.append(contact_features(seg, c, court, fps, prev_frame))
        prev_frame = c.frame
    return np.vstack(rows) if rows else np.zeros((0, N_FEATURES), dtype=np.float32)


LABELS = ("bounce", "hit", "other")
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}
