"""Synthetic match renderer + ground-truth generator.

Why simulate?  Real labelled match video is expensive, and a course project has
weeks, not months.  A physically-plausible renderer gives three things that raw
video does not:

* **Ground truth.**  Every bounce position, striker and rally winner is known
  exactly, so perception accuracy and rule accuracy can be measured separately
  (see :mod:`tools.benchmark`).
* **Regression tests.**  The same clip is generated deterministically, so the
  full pipeline can be tested in CI without shipping a 200 MB video.
* **A demo that always works** in a viva, on a laptop, offline.

The renderer draws a perspective view of a real-sized court using the same
canonical metre frame the rules use, so a rendered clip exercises the exact
code path used on real footage (homography, tracking, bounce detection, rules).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import apply_homography, homography_dlt
from ..core.types import Event, EventType, Point, Side
from ..court.model import CourtModel, CourtSpec

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

G = 9.81


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
@dataclass
class CameraView:
    """Pinhole-ish camera expressed as a court->image homography plus a height map."""

    H: np.ndarray                  # court metres -> image pixels (floor plane)
    width: int = 1280
    height: int = 720
    px_per_m_vertical: float = 46.0   # how much 1 m of height shifts the ball up

    @classmethod
    def perspective(cls, spec: CourtSpec, width: int = 1280, height: int = 720,
                    far_frac: float = 0.34, top: float = 0.14,
                    bottom: float = 0.95) -> "CameraView":
        """Elevated camera behind one baseline (a broadcast-style view).

        The far baseline appears narrow and high, the near baseline wide and low,
        which is the geometry in which racket-sport video is normally shot — and
        the geometry in which a ball's height shows up as vertical image motion.
        """
        far = far_frac / 2.0
        quad = {
            "tl": (width * (0.5 - far), height * top),
            "tr": (width * (0.5 + far), height * top),
            "br": (width * 0.985, height * bottom),
            "bl": (width * 0.015, height * bottom),
        }
        # court frame: x along the length (away from the camera), y across the width
        src = [(0.0, 0.0), (0.0, spec.width), (spec.length, spec.width), (spec.length, 0.0)]
        dst = [quad["tl"], quad["tr"], quad["br"], quad["bl"]]
        H = homography_dlt(src, dst)
        return cls(H=H, width=width, height=height,
                   px_per_m_vertical=float(height) * 0.05)

    def project(self, court_xy: Point, z_m: float = 0.0) -> Tuple[float, float]:
        p = apply_homography(self.H, [court_xy])[0]
        if not np.all(np.isfinite(p)):
            return float("nan"), float("nan")
        # crude but monotonic height cue: higher objects appear further up
        depth_scale = 0.65 + 0.5 * (p[1] / self.height)
        return float(p[0]), float(p[1] - z_m * self.px_per_m_vertical * depth_scale)


# --------------------------------------------------------------------------- #
# Script model
# --------------------------------------------------------------------------- #
@dataclass
class Shot:
    """One flight: struck at ``start``, lands at ``land`` after ``T`` seconds."""

    t: float
    start: Tuple[float, float, float]     # court x, y, height at contact
    land: Tuple[float, float]             # landing point (metres, z = 0)
    T: float = 1.6
    struck_by: Side = Side.A
    kind: str = "shot"                    # "serve" | "shot"
    apex_m: Optional[float] = None        # if set, T is derived from the apex

    @property
    def flight_time(self) -> float:
        """Ballistic flight time, or the explicit ``T`` when no apex is given."""
        if self.apex_m is None:
            return self.T
        z0 = max(0.05, float(self.start[2]))
        apex = max(z0 + 0.05, float(self.apex_m))
        vz = np.sqrt(2.0 * G * (apex - z0))
        t_up = vz / G
        t_down = np.sqrt(2.0 * G * apex) / G
        return float(t_up + t_down)

    @property
    def t_land(self) -> float:
        return self.t + self.flight_time


@dataclass
class RallyScript:
    index: int
    shots: List[Shot]
    expected_winner: str
    cause: str
    server: str
    expected_score: str = ""


@dataclass
class MatchScript:
    sport: str
    spec: CourtSpec
    players: Tuple[str, str]
    fps: float
    rallies: List[RallyScript]
    gap_s: float = 2.2
    start_s: float = 1.0

    @property
    def duration(self) -> float:
        if not self.rallies:
            return self.start_s + self.gap_s
        last = self.rallies[-1]
        end = last.shots[-1].t_land if last.shots else self.start_s
        return end + self.gap_s


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def sample_flight(shot: Shot, fps: float) -> List[Tuple[int, float, np.ndarray]]:
    """Sample one flight at ``fps``; returns (frame, t, court xyz)."""
    out = []
    T = shot.flight_time
    n = max(2, int(round(T * fps)))
    x0, y0, z0 = shot.start
    x1, y1 = shot.land
    if shot.apex_m is not None:
        vz0 = float(np.sqrt(2.0 * G * (max(z0 + 0.05, shot.apex_m) - z0)))
    else:
        vz0 = (-z0 + 0.5 * G * T ** 2) / T
    for k in range(n + 1):
        tau = T * k / n
        z = z0 + vz0 * tau - 0.5 * G * tau ** 2
        frac = k / n
        x = x0 + (x1 - x0) * frac
        y = y0 + (y1 - y0) * frac
        t = shot.t + tau
        out.append((int(round(t * fps)), float(t),
                    np.array([x, y, max(0.0, z)], dtype=np.float64)))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class SceneRenderer:
    """Draws court, net, players and ball for a scripted match."""

    COURT_BG = (58, 92, 46)
    COURT_FILL = (52, 108, 62)
    LINE = (245, 245, 245)
    NET = (35, 60, 120)

    def __init__(self, script: MatchScript, camera: Optional[CameraView] = None,
                 noise: bool = False, player_sway: bool = False):
        """``noise``/``player_sway`` enable a stress mode that mimics a real
        camera (sensor noise, moving players).  The default is a *clean* render
        because the demo exists to validate the scoring chain: motion-based
        detection is by construction unambiguous when the ball is the only
        moving object.  Real footage robustness comes from the trained detector
        path (see docs/TRAINING.md)."""
        self.script = script
        self.noise = bool(noise)
        self.player_sway = bool(player_sway)
        self.spec = script.spec
        self.cam = camera or CameraView.perspective(script.spec)
        self.court = CourtModel(script.spec, camera_court_homography(self.cam))
        self.samples: Dict[int, Tuple[float, np.ndarray]] = {}
        self._build_samples()

    # -- timeline -----------------------------------------------------------
    def _build_samples(self) -> None:
        fps = self.script.fps
        for rally in self.script.rallies:
            for shot in rally.shots:
                for frame, t, xyz in sample_flight(shot, fps):
                    self.samples[frame] = (t, xyz)
            # the ball rests at the final landing point briefly, then vanishes
            if rally.shots:
                last = rally.shots[-1]
                for k in range(1, int(0.4 * fps)):
                    f = int(round((last.t_land + k / fps) * fps))
                    self.samples.setdefault(f, (last.t_land + k / fps,
                                                np.array([last.land[0], last.land[1], 0.0])))

    def ball_at(self, t: float) -> Optional[np.ndarray]:
        frame = int(round(t * self.script.fps))
        return self.samples.get(frame, (None, None))[1]

    # -- drawing ------------------------------------------------------------
    def _draw_court(self, img: np.ndarray) -> None:
        if cv2 is None:  # pragma: no cover
            return
        spec = self.spec
        quad = self.cam.H
        # court fill
        corners = [(0, 0), (spec.length, 0), (spec.length, spec.width), (0, spec.width)]
        pts = np.array([self.cam.project(c) for c in corners], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, self.COURT_FILL)
        segs = [(p, q) for p, q in _court_segments(spec)]
        for a, b in segs:
            pa = self.cam.project(a)
            pb = self.cam.project(b)
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     self.LINE, 2, cv2.LINE_AA)
        # net posts and tape
        for y in (0.0, spec.width):
            top = self.cam.project((spec.net_x, y), spec.net_height_m)
            bot = self.cam.project((spec.net_x, y), 0.0)
            cv2.line(img, (int(top[0]), int(top[1])), (int(bot[0]), int(bot[1])),
                     (200, 200, 200), 4, cv2.LINE_AA)
        tape_a = self.cam.project((spec.net_x, 0.0), spec.net_height_m)
        tape_b = self.cam.project((spec.net_x, spec.width), spec.net_height_m)
        cv2.line(img, (int(tape_a[0]), int(tape_a[1])),
                 (int(tape_b[0]), int(tape_b[1])), self.NET, 3, cv2.LINE_AA)

    def _draw_players(self, img: np.ndarray, t: float) -> None:
        if cv2 is None:  # pragma: no cover
            return
        spec = self.spec
        for i, x in enumerate((spec.length * 0.16, spec.length * 0.84)):
            sway = 0.5 * np.sin(2 * np.pi * t / 3.0 + i) if self.player_sway else 0.0
            base = (x, spec.mid_y + sway)
            foot = self.cam.project(base, 0.0)
            head = self.cam.project(base, 1.75)
            col = (60, 60, 190) if i == 0 else (190, 110, 40)
            cv2.line(img, (int(foot[0]), int(foot[1])), (int(head[0]), int(head[1])),
                     col, 12, cv2.LINE_AA)
            cv2.circle(img, (int(head[0]), int(head[1])), 9, (200, 190, 170), -1, cv2.LINE_AA)

    def _draw_ball(self, img: np.ndarray, xyz: np.ndarray) -> None:
        if cv2 is None:  # pragma: no cover
            return
        shadow = self.cam.project((xyz[0], xyz[1]), 0.0)
        p = self.cam.project((xyz[0], xyz[1]), float(xyz[2]))
        if not (np.isfinite(p).all() and np.isfinite(shadow).all()):
            return
        cv2.circle(img, (int(shadow[0]), int(shadow[1])), 3, (30, 60, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), 7, (255, 255, 255), 1, cv2.LINE_AA)

    def frame(self, t: float) -> np.ndarray:
        img = np.full((self.cam.height, self.cam.width, 3), self.COURT_BG, np.uint8)
        self._draw_court(img)
        self._draw_players(img, t)
        xyz = self.ball_at(t)
        if xyz is not None:
            self._draw_ball(img, xyz)
        if self.noise and cv2 is not None:
            rng = np.random.default_rng(int(t * 1000) % 2 ** 31)
            noise = rng.normal(0, 2.0, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    # -- video --------------------------------------------------------------
    def write_video(self, path: str) -> str:
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv required to render a demo video")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.script.fps,
                                (self.cam.width, self.cam.height))
        n = int(round(self.script.duration * self.script.fps))
        for i in range(n):
            writer.write(self.frame(i / self.script.fps))
        writer.release()
        return path


def _court_segments(spec: CourtSpec) -> List[Tuple[Point, Point]]:
    s = spec
    segs = [
        ((0, 0), (s.length, 0)), ((0, s.width), (s.length, s.width)),
        ((0, 0), (0, s.width)), ((s.length, 0), (s.length, s.width)),
        ((0, s.singles_y_min), (s.length, s.singles_y_min)),
        ((0, s.singles_y_max), (s.length, s.singles_y_max)),
        ((s.net_x, 0), (s.net_x, s.width)),
    ]
    for x in s.service_lines_x:
        segs.append(((x, s.singles_y_min), (x, s.singles_y_max)))
    if s.has_centre_line:
        segs += [((0, s.mid_y), (s.service_lines_x[0], s.mid_y)),
                 ((s.service_lines_x[1], s.mid_y), (s.length, s.mid_y))]
    return segs


def camera_court_homography(cam: CameraView) -> np.ndarray:
    """Homography that maps image pixels -> court metres (what the analyzer wants)."""
    return np.linalg.inv(cam.H)


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def perfect_events(script: MatchScript) -> List[Event]:
    """Ideal perception output: exact contacts and landings, no noise.

    Running the rally engine on these events and comparing with the engine run
    on *detected* events isolates perception error from rule error.
    """
    events: List[Event] = []
    for rally in script.rallies:
        for i, shot in enumerate(rally.shots):
            # A shot that is intercepted in the air never touches the floor:
            # emitting its landing would invent a rally-ending contact.
            nxt = rally.shots[i + 1] if i + 1 < len(rally.shots) else None
            intercepted = nxt is not None and nxt.t < shot.t_land
            # a "dead bounce" is the ball bouncing out of play, not a stroke
            if shot.kind != "dead_bounce":
                etype = EventType.SERVE if shot.kind == "serve" else EventType.SHOT
                events.append(Event(
                    type=etype, frame=int(round(shot.t * script.fps)), t=shot.t,
                    side=shot.struck_by, court_xy=(shot.start[0], shot.start[1]),
                    img_xy=(0.0, 0.0), confidence=1.0,
                    reason=f"scripted {shot.kind}"))
            if intercepted:
                continue
            events.append(Event(
                type=EventType.LANDING,
                frame=int(round(shot.t_land * script.fps)), t=shot.t_land,
                side=Side.A if shot.land[0] < script.spec.net_x else Side.B,
                court_xy=shot.land, img_xy=(0.0, 0.0), confidence=1.0,
                reason="scripted landing"))
    events.sort(key=lambda e: e.frame)
    return events


def oracle_track(script: MatchScript, camera: Optional[CameraView] = None,
                 jitter_px: float = 0.0, dropout_prob: float = 0.0,
                 seed: int = 0) -> List[TrackPoint]:
    """The ball trajectory a *perfect* tracker would report.

    Used to validate everything downstream of detection (event semantics, rules,
    scoring, evidence) on real video frames, independently of detector quality.
    ``jitter_px`` and ``dropout_prob`` inject controlled noise so the robustness
    of the event layer can be measured as a separate curve.
    """
    from ..core.types import TrackPoint

    cam = camera or CameraView.perspective(script.spec)
    renderer = SceneRenderer(script, cam)
    rng = np.random.default_rng(seed)
    out: List[TrackPoint] = []
    for frame in sorted(renderer.samples):
        t, xyz = renderer.samples[frame]
        img = cam.project((xyz[0], xyz[1]), float(xyz[2]))
        if not np.isfinite(img[0]) or not np.isfinite(img[1]):
            continue
        if 0 <= img[0] < cam.width and 0 <= img[1] < cam.height:
            pass
        else:
            continue                      # outside the frame: a detector could not see it
        if dropout_prob > 0 and rng.random() < dropout_prob:
            continue
        xy = (img[0] + rng.normal(0, jitter_px), img[1] + rng.normal(0, jitter_px))
        out.append(TrackPoint(frame=frame, t=float(t), xy=xy, conf=1.0))
    return out


def script_to_dict(script: MatchScript) -> Dict[str, Any]:
    return {
        "sport": script.sport,
        "players": list(script.players),
        "fps": script.fps,
        "duration_s": round(script.duration, 2),
        "rallies": [
            {
                "index": r.index,
                "server": r.server,
                "expected_winner": r.expected_winner,
                "cause": r.cause,
                "expected_score": r.expected_score,
                "shots": [{"t": s.t, "start": list(s.start), "land": list(s.land),
                           "T": s.T, "by": s.struck_by.value, "kind": s.kind}
                          for s in r.shots],
            }
            for r in script.rallies
        ],
    }


def write_ground_truth(script: MatchScript, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(script_to_dict(script), fh, indent=2)
    return path
