"""Drawing helpers for evidence images.

Everything is drawn with OpenCV primitives so a proof JPEG can be produced on
the edge device with no browser, no matplotlib and no fonts to install.  Text
sizes scale with the frame, so the same code produces readable crops for a
1080p clip and for a 4K one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Event, EventType, Point, ScoreState
from ..court.model import CourtModel

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

IN_COLOR = (80, 220, 80)       # BGR green
OUT_COLOR = (60, 60, 235)      # BGR red
LINE_COLOR = (0, 215, 255)     # amber for line calls
BALL_COLOR = (0, 255, 255)
COURT_COLOR = (255, 255, 255)
TEXT_BG = (32, 32, 32)


def _font_scale(w: int, base: float = 0.5) -> float:
    return max(0.35, base * w / 1280.0)


def draw_text_block(frame: np.ndarray, lines: Sequence[str],
                    origin: Tuple[int, int] = (12, 28),
                    font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255),
                    bg: Optional[Tuple[int, int, int]] = TEXT_BG, alpha: float = 0.65,
                    line_height: float = 1.5, thickness: int = 1) -> np.ndarray:
    """Multi-line text with an optional translucent background panel."""
    if cv2 is None or not lines:
        return frame
    fs = _font_scale(frame.shape[1], font_scale)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)[0] for t in lines]
    pad = int(10 * fs / 0.5) if fs > 0 else 8
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = int(sum(s[1] for s in sizes) + (len(lines) - 1) * int(12 * fs / 0.5) + 2 * pad)
    x, y = origin

    if bg is not None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - pad // 2, y - sizes[0][1] - pad // 2),
                      (x + box_w, y + box_h), bg, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cy = y
    for text, (tw, th) in zip(lines, sizes):
        cv2.putText(frame, text, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, color,
                    thickness, cv2.LINE_AA)
        cy += th + int(12 * fs / 0.5)
    return frame


def draw_court_overlay(frame: np.ndarray, court: CourtModel,
                       mode: str = "singles",
                       color: Tuple[int, int, int] = COURT_COLOR,
                       thickness: int = 2) -> np.ndarray:
    """Project the court lines and the play polygon into the image."""
    if cv2 is None or not court.registered:
        return frame
    try:
        segs = court.line_segments(mode)
        pts = np.array([p for seg in segs for p in seg], dtype=np.float64)
        proj = court.to_image(pts)
        for i, (a, b) in enumerate(zip(proj[0::2], proj[1::2])):
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     color, thickness, cv2.LINE_AA)
        # net band
        net = court.to_image([(court.spec.net_x, 0.0), (court.spec.net_x, court.spec.width)])
        if np.all(np.isfinite(net)):
            cv2.line(frame, tuple(int(v) for v in net[0]), tuple(int(v) for v in net[1]),
                     (255, 120, 0), thickness + 1, cv2.LINE_AA)
    except Exception:
        return frame
    return frame


def draw_ball(frame: np.ndarray, xy: Point, conf: float = 1.0,
              color: Tuple[int, int, int] = BALL_COLOR, radius: int = 7) -> None:
    if cv2 is None or xy is None:
        return
    r = int(max(4, radius * frame.shape[1] / 1280.0))
    c = tuple(int(v) for v in xy)
    cv2.circle(frame, c, r + 3, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, c, r, color, -1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, points: Sequence[Point],
               color_in: Tuple[int, int, int] = IN_COLOR,
               color_out: Tuple[int, int, int] = OUT_COLOR,
               thickness: int = 3, fade: bool = True) -> None:
    """Polyline through the ball's recent positions, fading with age."""
    if cv2 is None or len(points) < 2:
        return
    pts = np.array(points, dtype=np.float64)
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        alpha = (i / len(pts)) if fade else 1.0
        col = tuple(int(c * (0.35 + 0.65 * alpha)) for c in color_in)
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col,
                 thickness, cv2.LINE_AA)


def draw_landing_marker(frame: np.ndarray, img_xy: Point, in_bounds: bool,
                        label: str, value_text: str = "", radius: int = 26) -> None:
    """Ring + crosshair on the landing point, coloured by the call."""
    if cv2 is None or img_xy is None:
        return
    color = IN_COLOR if in_bounds else OUT_COLOR
    c = (int(img_xy[0]), int(img_xy[1]))
    r = int(max(12, radius * frame.shape[1] / 1280.0))
    th = max(2, r // 8)
    cv2.circle(frame, c, r, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.circle(frame, c, r, color, th, cv2.LINE_AA)
    cv2.line(frame, (c[0] - r, c[1]), (c[0] + r, c[1]), color, th, cv2.LINE_AA)
    cv2.line(frame, (c[0], c[1] - r), (c[0], c[1] + r), color, th, cv2.LINE_AA)
    fs = _font_scale(frame.shape[1], 0.7)
    anchor = (c[0] + r + 8, c[1] - 6)
    for i, text in enumerate(t for t in (label, value_text) if t):
        y = anchor[1] + i * int(26 * fs / 0.7 + 6)
        cv2.putText(frame, text, (anchor[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), int(fs * 4) + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (anchor[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    color, max(1, int(fs * 2.5)), cv2.LINE_AA)


def draw_zoom_inset(frame: np.ndarray, img_xy: Point, src_frame: np.ndarray,
                    size_px: int = 190, zoom: int = 3,
                    label: str = "landing zoom") -> np.ndarray:
    """Zoomed crop around the landing, pasted bottom-right (the 'hawk-eye' pane)."""
    if cv2 is None or img_xy is None:
        return frame
    h, w = src_frame.shape[:2]
    half = size_px // (2 * zoom)
    x0 = int(np.clip(img_xy[0] - half, 0, w - 1))
    y0 = int(np.clip(img_xy[1] - half, 0, h - 1))
    x1 = int(np.clip(img_xy[0] + half, x0 + 2, w))
    y1 = int(np.clip(img_xy[1] + half, y0 + 2, h))
    crop = src_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    big = cv2.resize(crop, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    # mark the exact point in the zoom pane
    zx = int((img_xy[0] - x0) / max(1e-6, (x1 - x0)) * size_px)
    zy = int((img_xy[1] - y0) / max(1e-6, (y1 - y0)) * size_px)
    cv2.circle(big, (zx, zy), 6, (0, 255, 255), 2, cv2.LINE_AA)
    pad = 12
    fx1, fy1 = w - pad, h - pad
    fx0, fy0 = fx1 - size_px, fy1 - size_px
    if fx0 > 0 and fy0 > 0:
        frame[fy0:fy1, fx0:fx1] = big
        cv2.rectangle(frame, (fx0, fy0), (fx1, fy1), (255, 255, 255), 2)
        cv2.putText(frame, label, (fx0 + 6, fy0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_event_tags(frame: np.ndarray, events: Sequence[Event],
                    origin: Tuple[int, int] = (12, 120)) -> np.ndarray:
    """Compact list of the events behind the current call."""
    lines = []
    seen = set()
    for ev in events:
        label = ev.type.value.replace("_", " ").upper()
        extra = ""
        if ev.court_xy is not None:
            extra = f"  @({ev.court_xy[0]:.2f},{ev.court_xy[1]:.2f})m"
        # the detector can emit the same physical event twice (two tracklets
        # ending on the same contact); the proof image should read cleanly
        key = (label, extra)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{label}{extra}")
    lines = lines[-7:]
    if lines:
        return draw_text_block(frame, lines, origin=origin, font_scale=0.48,
                               color=(235, 235, 235))
    return frame


# --------------------------------------------------------------------------- #
# Scoreboard render
# --------------------------------------------------------------------------- #
def render_scoreboard(state: ScoreState, players: Sequence[str],
                      width: int = 900, height: int = 260,
                      title: str = "") -> np.ndarray:
    """Standalone scoreboard image (dark card with the current score)."""
    if cv2 is None:  # pragma: no cover
        return np.zeros((height, width, 3), np.uint8)
    img = np.full((height, width, 3), 26, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (70, 70, 70), 2)
    cv2.putText(img, (title or state.sport.value.upper())[:48], (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

    score = state.score or {}
    sets = score.get("set_scores", [])
    games = score.get("games", {})
    points = score.get("points", {})
    games_won = score.get("games_won", score.get("sets_won", {}))

    y = 96
    for name in players:
        is_server = name == state.serving_player
        col = (120, 255, 120) if is_server else (230, 230, 230)
        marker = "\u25cf" if is_server else " "
        cv2.putText(img, f"{marker} {name}", (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, col, 2, cv2.LINE_AA)
        x = int(width * 0.45)
        for i, s in enumerate(sets):
            idx = players.index(name) if name in players else 0
            val = s[idx] if idx < len(s) else 0
            cv2.putText(img, str(val), (x + i * 60, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, str(games.get(name, 0)), (int(width * 0.78), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 210, 120), 2, cv2.LINE_AA)
        ptxt = str(points.get(name, ""))
        if score.get("points_display"):
            ptxt = score["points_display"].split("-")[
                0 if players.index(name) == 0 else 1] if "-" in score["points_display"] else ptxt
        cv2.putText(img, ptxt, (int(width * 0.88), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (120, 220, 255), 2, cv2.LINE_AA)
        y += 56

    cv2.putText(img, state.display[:70], (24, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"t={state.t:7.2f}s  frame={state.frame}", (int(width * 0.55), height - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def render_summary(result: dict, width: int = 1100, row_h: int = 30) -> np.ndarray:
    """Tabular proof sheet: every point with time, winner, cause and score."""
    if cv2 is None:  # pragma: no cover
        return np.zeros((200, width, 3), np.uint8)
    points = result.get("points", [])
    height = 140 + row_h * max(1, len(points))
    img = np.full((height, width, 3), 24, np.uint8)
    cv2.putText(img, "SMARTGYM MATCH PROOF SHEET", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    meta = result.get("meta", {})
    sub = (f"{str(meta.get('sport','')).upper()}  |  {meta.get('players','')}  |  "
           f"video: {meta.get('video','')}")
    cv2.putText(img, sub[:110], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"final score: {result.get('final_score','')}   "
                     f"winner: {result.get('winner') or 'n/a'}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)

    headers = ["#", "t (s)", "winner", "cause", "score after", "review"]
    xs = [20, 70, 150, 300, 620, 940]
    for x, htext in zip(xs, headers):
        cv2.putText(img, htext, (x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
    y = 130 + row_h
    for p in points:
        t_end = p.get("t_end", 0.0)
        vals = [str(p.get("index", "")), f"{t_end:.2f}",
                str(p.get("winner", ""))[:14], str(p.get("cause", ""))[:26],
                str((p.get("score_after") or {}).get("display", ""))[:28],
                "yes" if p.get("needs_review") else ""]
        for x, v in zip(xs, vals):
            cv2.putText(img, v, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (225, 225, 225), 1, cv2.LINE_AA)
        y += row_h
    return img
