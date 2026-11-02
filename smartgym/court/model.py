"""Court geometry: canonical metric frame, homography registration, line calls.

Canonical court coordinates (metres)
------------------------------------
::

    y
    ^
    |  (0,0) ................ (L,0)      <- baseline A / baseline B
    |    |                     |
    |    |        NET          |          x runs along the court length
    |    |         |           |
    |  (0,W) ................ (L,W)
    +------------------------------> x

Side ``A`` owns ``x < L/2``, side ``B`` owns ``x > L/2``.  All rules in
``smartgym.sports`` are written against this frame, which makes them
independent of camera placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import (
    apply_homography,
    apply_homography_single,
    distance_to_segments,
    homography_dlt,
    point_in_polygon,
    polygon_mask,
)
from ..core.types import Point, Side


@dataclass
class CourtSpec:
    """Official dimensions of a court (metres)."""

    sport: str
    length: float
    width: float                 # doubles / full width
    singles_width: float
    # distance from the net to the service line
    service_line_from_net: float
    # badminton: doubles long service line is inset from the back boundary.
    # tennis: 0.0 (the baseline is the back limit).
    doubles_long_service_inset: float = 0.0
    has_centre_line: bool = True
    net_height_m: float = 1.55
    # badminton: in doubles the serve must stop at the doubles long service
    # line (5.94 m from the net); tennis serves always stop at 6.40 m.
    doubles_serve_uses_long_service_line: bool = True
    # badminton doubles serves may use the full width; tennis serves never do
    serve_uses_full_width_in_doubles: bool = True

    # -- derived landmarks --------------------------------------------------
    @property
    def net_x(self) -> float:
        return self.length / 2.0

    @property
    def mid_y(self) -> float:
        return self.width / 2.0

    @property
    def singles_y_min(self) -> float:
        return (self.width - self.singles_width) / 2.0

    @property
    def singles_y_max(self) -> float:
        return self.width - self.singles_y_min

    @property
    def service_lines_x(self) -> Tuple[float, float]:
        return (self.net_x - self.service_line_from_net,
                self.net_x + self.service_line_from_net)

    @property
    def long_service_x(self) -> Tuple[float, float]:
        inset = self.doubles_long_service_inset
        return (inset, self.length - inset)

    def sample_landmarks(self) -> List[Tuple[str, Point]]:
        """Named points usable for calibration clicks / auto-detection."""
        lm = [
            ("corner_0", (0.0, 0.0)),
            ("corner_1", (self.length, 0.0)),
            ("corner_2", (self.length, self.width)),
            ("corner_3", (0.0, self.width)),
            ("net_left", (self.net_x, 0.0)),
            ("net_right", (self.net_x, self.width)),
        ]
        for i, x in enumerate(self.service_lines_x):
            lm.append((f"service_line_{i}_side0", (x, self.singles_y_min)))
            lm.append((f"service_line_{i}_side1", (x, self.singles_y_max)))
        lm.append(("centre_net", (self.net_x, self.mid_y)))
        return lm


# --------------------------------------------------------------------------- #
# Presets (official dimensions)
# --------------------------------------------------------------------------- #
BADMINTON = CourtSpec(
    sport="badminton",
    length=13.40,          # BWF: 13.4 m
    width=6.10,            # doubles width
    singles_width=5.18,    # singles width
    service_line_from_net=1.98,       # short service line
    doubles_long_service_inset=0.76,  # doubles long service line from baseline
    has_centre_line=True,
    net_height_m=1.55,     # at posts; 1.524 at centre
    doubles_serve_uses_long_service_line=True,
    serve_uses_full_width_in_doubles=True,
)

TENNIS = CourtSpec(
    sport="tennis",
    length=23.77,          # ITF baseline to baseline
    width=10.97,           # doubles
    singles_width=8.23,
    service_line_from_net=6.40,
    doubles_long_service_inset=0.0,
    has_centre_line=True,
    net_height_m=0.914,    # centre strap
    doubles_serve_uses_long_service_line=False,   # serve box is always 6.40 m
    serve_uses_full_width_in_doubles=False,       # alley is OUT on a serve
)

VOLLEYBALL = CourtSpec(  # 18 x 9 m, attack line 3 m from net (extension point)
    sport="volleyball",
    length=18.0,
    width=9.0,
    singles_width=9.0,
    service_line_from_net=9.0,
    net_height_m=2.43,
)

PRESETS: Dict[str, CourtSpec] = {s.sport: s for s in (BADMINTON, TENNIS, VOLLEYBALL)}


# --------------------------------------------------------------------------- #
# Court model
# --------------------------------------------------------------------------- #
class CourtModel:
    """Registered court: maps pixels <-> metres and answers in/out questions."""

    def __init__(self, spec: CourtSpec, homography: Optional[np.ndarray] = None):
        self.spec = spec
        self._H: Optional[np.ndarray] = None
        self._H_inv: Optional[np.ndarray] = None
        if homography is not None:
            self.set_homography(homography)

    # -- registration -------------------------------------------------------
    #: canonical corner order used for manual calibration (image TL,TR,BR,BL)
    CANONICAL_CORNERS = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))

    @classmethod
    def from_corners(cls, spec: CourtSpec, corners_img: Sequence[Point],
                     order: str = "tl_tr_br_bl") -> "CourtModel":
        """Calibrate from the 4 outer court corners.

        Corner convention (``order="tl_tr_br_bl"``, the default): give the four
        corners **clockwise in the image, starting at the top-left**, i.e. the
        far-left corner of the court as it appears on screen.  They are mapped to
        court coordinates ``(0,0) -> (0,W) -> (L,W) -> (L,0)``:

        * the far baseline (top of the image) is ``x = 0``,
        * ``+x`` runs towards the camera,
        * ``+y`` runs to the right of the image.

        With that convention a player standing in the far half (side ``A``) and
        facing the camera has their right-hand service court at large ``y``, which
        is exactly what the serving rules in :mod:`smartgym.sports` assume.

        ``order="length_first"`` maps to ``(0,0),(L,0),(L,W),(0,W)`` instead, for
        cameras where the court length runs left-to-right across the image.
        """
        if len(corners_img) != 4:
            raise ValueError("exactly 4 corners required")
        if order == "length_first":
            court_pts = [(0.0, 0.0), (spec.length, 0.0),
                         (spec.length, spec.width), (0.0, spec.width)]
        else:
            court_pts = [(0.0, 0.0), (0.0, spec.width),
                         (spec.length, spec.width), (spec.length, 0.0)]
        H = homography_dlt(corners_img, court_pts)
        return cls(spec, H)

    @classmethod
    def from_correspondences(cls, spec: CourtSpec,
                             pairs: Sequence[Tuple[Point, Point]]) -> "CourtModel":
        """Calibrate from any >= 4 (image, court-metres) pairs."""
        src = [p[0] for p in pairs]
        dst = [p[1] for p in pairs]
        return cls(spec, homography_dlt(src, dst))

    def set_homography(self, H: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        self._H = H
        self._H_inv = np.linalg.inv(H)

    @property
    def registered(self) -> bool:
        return self._H is not None

    @property
    def homography(self) -> np.ndarray:
        if self._H is None:
            raise RuntimeError("court not calibrated")
        return self._H

    def to_court(self, pts: Sequence[Point]) -> np.ndarray:
        return apply_homography(self.homography, pts)

    def to_court_point(self, pt: Point) -> Optional[Point]:
        if not self.registered:
            return None
        return apply_homography_single(self.homography, pt)

    def to_image(self, pts: Sequence[Point]) -> np.ndarray:
        if self._H_inv is None:
            raise RuntimeError("court not calibrated")
        return apply_homography(self._H_inv, pts)

    # -- frame --------------------------------------------------------------
    def frame_polygon(self, margin: float = 0.0) -> List[Point]:
        """Outer doubles rectangle in canonical metres, optionally inflated."""
        s = self.spec
        return [(-margin, -margin), (s.length + margin, -margin),
                (s.length + margin, s.width + margin), (-margin, s.width + margin)]

    def frame_polygon_img(self, margin: float = 0.0) -> np.ndarray:
        if not self.registered:
            raise RuntimeError("court not calibrated")
        return self.to_image(self.frame_polygon(margin))

    # -- line calls ---------------------------------------------------------
    def play_polygon(self, mode: str = "singles") -> List[Point]:
        """Polygon the ball must land inside to be IN (rally context)."""
        s = self.spec
        if mode == "doubles" or s.singles_width >= s.width:
            y0, y1 = 0.0, s.width
        else:
            y0, y1 = s.singles_y_min, s.singles_y_max
        return [(0.0, y0), (s.length, y0), (s.length, y1), (0.0, y1)]

    def serve_depth(self, mode: str = "singles") -> float:
        """Distance from the net to the back limit of the service box."""
        s = self.spec
        if mode == "doubles" and s.doubles_serve_uses_long_service_line:
            return (s.length / 2.0) - s.doubles_long_service_inset
        return s.service_line_from_net

    def serve_width_limits(self, mode: str = "singles") -> Tuple[float, float]:
        """Lateral limits of the service box (the alley is out on a tennis serve)."""
        s = self.spec
        if mode == "doubles" and s.serve_uses_full_width_in_doubles:
            return 0.0, s.width
        return s.singles_y_min, s.singles_y_max

    def service_box_for_band(self, target_side: Side, target_low_band: bool,
                            mode: str = "singles") -> List[Point]:
        """Service box on ``target_side`` for a serve aimed at one y-band.

        ``target_low_band`` means the box lives in the ``y < width/2`` half.
        """
        s = self.spec
        y_out_lo, y_out_hi = self.serve_width_limits(mode)
        y0, y1 = (y_out_lo, s.mid_y) if target_low_band else (s.mid_y, y_out_hi)
        depth = self.serve_depth(mode)
        if target_side is Side.B:
            x0, x1 = s.net_x, s.net_x + depth
        else:
            x0, x1 = s.net_x - depth, s.net_x
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def service_box(self, server_pt: Point, mode: str = "singles") -> List[Point]:
        """Diagonal service box for a server standing at ``server_pt``."""
        s = self.spec
        target_side = self.half_of(server_pt).other
        target_low_band = not (server_pt[1] < s.mid_y)  # cross-court: opposite band
        return self.service_box_for_band(target_side, target_low_band, mode)

    def in_play(self, pt: Point, mode: str = "singles", margin: float = 0.0) -> bool:
        """True if ``pt`` (court metres) is inside the playing area.

        ``margin`` expands the court by the ball radius so that a ball whose
        centre is a centimetre outside a line still counts as IN — this is how
        line judges (and hawk-eye) treat the ball's edge, not its centre.
        """
        return point_in_polygon(self._inflate(self.play_polygon(mode), margin), pt)

    def in_service_box(self, pt: Point, box: Sequence[Point], margin: float = 0.0) -> bool:
        return point_in_polygon(self._inflate(box, margin), pt)

    @staticmethod
    def _inflate(poly: Sequence[Point], margin: float) -> List[Point]:
        if margin <= 0:
            return list(poly)
        pts = np.asarray(poly, dtype=np.float64)
        c = pts.mean(axis=0)
        out = []
        for p in pts:
            d = p - c
            n = np.linalg.norm(d)
            out.append(tuple(p + (d / n * margin if n > 1e-9 else 0.0)))
        return out

    def line_segments(self, mode: str = "singles") -> List[Tuple[Point, Point]]:
        s = self.spec
        y0, y1 = (0.0, s.width) if mode == "doubles" else (s.singles_y_min, s.singles_y_max)
        segs: List[Tuple[Point, Point]] = [
            ((0.0, y0), (0.0, y1)),
            ((s.length, y0), (s.length, y1)),
            ((0.0, y0), (s.length, y0)),
            ((0.0, y1), (s.length, y1)),
            ((s.net_x, 0.0), (s.net_x, s.width)),
        ]
        for x in s.service_lines_x:
            segs.append(((x, y0), (x, y1)))
        if s.has_centre_line:
            segs += [((0.0, s.mid_y), (s.service_lines_x[0], s.mid_y)),
                     ((s.service_lines_x[1], s.mid_y), (s.length, s.mid_y))]
        return segs

    def distance_to_nearest_line(self, pt: Point, mode: str = "singles") -> float:
        return distance_to_segments(pt, self.line_segments(mode))

    def half_of(self, pt: Point) -> Side:
        return Side.A if pt[0] < self.spec.net_x else Side.B

    def side_polygon(self, side: Side, mode: str = "singles") -> List[Point]:
        s = self.spec
        y0, y1 = (0.0, s.width) if mode == "doubles" else (s.singles_y_min, s.singles_y_max)
        if side is Side.A:
            return [(0.0, y0), (s.net_x, y0), (s.net_x, y1), (0.0, y1)]
        return [(s.net_x, y0), (s.length, y0), (s.length, y1), (s.net_x, y1)]

    def meta(self) -> dict:
        s = self.spec
        return {
            "sport": s.sport,
            "length_m": s.length,
            "width_m": s.width,
            "singles_width_m": s.singles_width,
            "net_x_m": s.net_x,
            "registered": self.registered,
            "homography": None if self._H is None else self._H.tolist(),
        }
