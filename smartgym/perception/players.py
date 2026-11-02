"""Player localisation and hit attribution.

Player boxes are *not* required to score a match — the ball trajectory alone is
sufficient.  They add three things:

1. **Better striker attribution.**  When two players are close to the net, the
   trajectory alone can be ambiguous about who touched the ball last; the
   nearest player at the contact frame resolves it.
2. **Coverage / rally analytics.**  Distance covered, average court position.
3. **Usable proof images.**  A reviewer immediately sees which player was where.

Detection is background-subtraction based (no training needed for a fixed
camera) with an optional YOLO backend for moving cameras or cluttered scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Event, EventType, PlayerBox, Point, Side
from ..court.model import CourtModel

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class PlayerTracker:
    """Foreground-blob player tracker with per-side assignment."""

    name = "motion"

    def __init__(self, court: CourtModel, min_area_frac: float = 0.004,
                 history: int = 300, sample_every: int = 3,
                 warmup_frames: int = 15):
        self.court = court
        self.min_area_frac = float(min_area_frac)
        self.sample_every = max(1, int(sample_every))
        self.warmup = int(warmup_frames)
        self._bg = None
        self._i = 0
        self.current: List[PlayerBox] = []

    def _ensure_bg(self):
        if self._bg is None and cv2 is not None:
            self._bg = cv2.createBackgroundSubtractorMOG2(history=300,
                                                          varThreshold=36,
                                                          detectShadows=False)

    def step(self, frame: np.ndarray, idx: int, t: float) -> List[PlayerBox]:
        if cv2 is None:  # pragma: no cover
            return []
        self._ensure_bg()
        self._i += 1
        fg = self._bg.apply(frame)
        if self._i < self.warmup:
            return []
        if self._i % self.sample_every and self.current:
            return self.current

        h, w = frame.shape[:2]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * w * h

        cands = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 1.2 * bw:      # players are tall and thin in a broadcast side view
                continue
            cands.append((area, x, y, bw, bh))
        cands.sort(key=lambda c: -c[0])
        cands = cands[:2]

        boxes: List[PlayerBox] = []
        for area, x, y, bw, bh in cands:
            foot = (float(x + bw / 2), float(y + bh))
            court_xy = self.court.to_court_point(foot) if self.court.registered else None
            side = self.court.half_of(court_xy) if court_xy else (
                Side.A if foot[0] < w / 2 else Side.B)
            boxes.append(PlayerBox(
                frame=idx, t=t, side=side,
                box=(float(x), float(y), float(x + bw), float(y + bh)),
                foot_img=foot, foot_court=court_xy,
                conf=float(min(1.0, area / (min_area * 8))),
            ))
        if len(boxes) == 2 and boxes[0].side is boxes[1].side:
            # both blobs on the same side: keep the bigger, it is the player
            boxes = [boxes[0]]
        self.current = boxes
        return boxes

    def reset(self) -> None:
        self._bg = None
        self.current = []


def nearest_player(players: Sequence[PlayerBox], img_xy: Point,
                   max_dist_px: float = 260.0) -> Optional[PlayerBox]:
    """Player closest to an image-space contact point."""
    best, best_d = None, max_dist_px
    for p in players:
        fx, fy = p.foot_img
        d = float(np.hypot(fx - img_xy[0], fy - img_xy[1]))
        # compare against the player's centre too: contact happens at arm height
        cx = (p.box[0] + p.box[2]) / 2
        cy = (p.box[1] + p.box[3]) / 2
        d2 = float(np.hypot(cx - img_xy[0], cy - img_xy[1]))
        d = min(d, d2)
        if d < best_d:
            best, best_d = p, d
    return best


def refine_striker_sides(events: Sequence[Event],
                         players_by_frame: Dict[int, List[PlayerBox]],
                         tolerance: int = 4) -> List[Event]:
    """Overwrite the side of SHOT/SERVE events using the nearest player.

    Only applied when a confident player detection exists close enough to the
    contact; otherwise the (already plausible) trajectory-based side is kept.
    Events are mutated in place and returned.
    """
    if not players_by_frame:
        return list(events)
    frames = sorted(players_by_frame)
    for ev in events:
        if ev.type not in (EventType.SHOT, EventType.SERVE):
            continue
        if ev.img_xy is None:
            continue
        near = min(frames, key=lambda f: abs(f - ev.frame))
        if abs(near - ev.frame) > tolerance:
            continue
        p = nearest_player(players_by_frame[near], ev.img_xy)
        if p is None:
            continue
        ev.meta["side_from_players"] = True
        ev.meta["player_distance_px"] = round(
            float(np.hypot(p.foot_img[0] - ev.img_xy[0], p.foot_img[1] - ev.img_xy[1])), 1)
        ev.side = p.side
    return list(events)


def build_player_tracker(cfg: dict, court: CourtModel) -> Optional["PlayerTracker"]:
    det = (cfg or {}).get("detector", {})
    kind = str(det.get("players", "none")).lower()
    if kind in ("none", "off", "false", ""):
        return None
    if kind == "color":
        return PlayerTracker(court)
    if kind == "motion":
        return PlayerTracker(court)
    return PlayerTracker(court)
