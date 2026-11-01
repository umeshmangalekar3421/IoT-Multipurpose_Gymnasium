"""Court calibration: manual corners, config homography, or auto line detection.

Everything the scoring logic does is expressed in court metres, so this file is
the single point where pixels become physics.  Three levels of effort:

1. **From config** — four corner pixels clicked once per camera setup.  Most
   accurate and what a fixed gym installation should use.
2. **From homography** — a 3x3 matrix measured offline (e.g. with a printed
   calibration target); no clicking at runtime.
3. **Auto** — detect the painted court lines in the first frames and fit the
   quadrilateral.  Works well for badminton mats and tennis hard courts with
   high-contrast lines; it is a convenience, not a replacement for (1).

The auto-calibrator also *validates* a calibration: a homography that maps the
court to a self-intersecting or absurdly small quadrilateral is rejected, which
prevents silently scoring a whole match in a garbage coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import apply_homography, homography_dlt, polygon_mask
from ..core.types import Point
from ..court.model import CourtModel, CourtSpec

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_homography(H: np.ndarray, spec: CourtSpec,
                        image_size: Tuple[int, int]) -> Tuple[bool, str]:
    """Sanity-check a homography by re-projecting the court into the image."""
    try:
        w, h = image_size
        corners_court = [(0.0, 0.0), (spec.length, 0.0),
                         (spec.length, spec.width), (0.0, spec.width)]
        Hcv = np.linalg.inv(np.asarray(H, dtype=np.float64))
        proj = apply_homography(Hcv, corners_court)
        if not np.all(np.isfinite(proj)):
            return False, "corner projection is not finite"
        xs, ys = proj[:, 0], proj[:, 1]
        if xs.max() - xs.min() < 0.15 * w or ys.max() - ys.min() < 0.15 * h:
            return False, "projected court is too small in the frame"
        if (xs.max() < 0 or xs.min() > w or ys.max() < 0 or ys.min() > h):
            return False, "projected court lies entirely outside the frame"
        if np.any(np.abs(proj) > 50 * max(w, h)):
            return False, "projection exploded (degenerate homography)"
        # self-intersection test on the projected quad
        if _self_intersects(proj):
            return False, "projected court quadrilateral self-intersects"
        return True, "ok"
    except Exception as exc:  # pragma: no cover
        return False, f"validation error: {exc}"


def _self_intersects(quad: np.ndarray) -> bool:
    def cross(a, b):
        return np.sign((b[0] - a[0]) * (b[1] - a[1]))

    def seg_intersect(p1, p2, p3, p4) -> bool:
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    n = len(quad)
    for i in range(n):
        a, b = quad[i], quad[(i + 1) % n]
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            c, d = quad[j], quad[(j + 1) % n]
            if seg_intersect(a, b, c, d):
                return True
    return False


def order_quad(points: Sequence[Point]) -> List[Point]:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("need 4 points")
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def court_line_alignment_cost(court: CourtModel, line_edges: np.ndarray,
                              mode: str = "doubles") -> float:
    """Mean distance (fraction of the image diagonal) from projected court
    lines to the nearest detected line pixel.

    This is the key to disambiguating a quadrilateral: the *length* axis and the
    *width* axis look alike, so a quad can be interpreted in four different ways
    (the court rotated by 0/90/180/270 degrees).  Only the correct interpretation
    makes the projected inner lines (service lines, centre line) land on real
    painted lines.
    """
    if cv2 is None:  # pragma: no cover
        return float("inf")
    if line_edges is None or line_edges.size == 0:
        return float("inf")
    h, w = line_edges.shape[:2]
    dist = cv2.distanceTransform(255 - line_edges, cv2.DIST_L2, 3)
    samples = []
    for a, b in court.line_segments(mode):
        n = max(4, int(np.hypot(b[0] - a[0], b[1] - a[1]) * 6))
        for k in range(n):
            f = k / (n - 1)
            samples.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    try:
        proj = court.to_image(samples)
    except Exception:
        return float("inf")
    ok = np.isfinite(proj).all(axis=1)
    if ok.sum() < 8:
        return float("inf")
    proj = proj[ok]
    inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w - 1) &
              (proj[:, 1] >= 0) & (proj[:, 1] < h - 1))
    if inside.mean() < 0.5:
        return float("inf")
    vals = dist[proj[inside, 1].astype(int), proj[inside, 0].astype(int)]
    return float(vals.mean() / np.hypot(w, h))


def _labeling_candidates(quad: Sequence[Point], spec: CourtSpec):
    """The eight ways a detected quad can be mapped onto the court rectangle.

    A quadrilateral does not say which image edge is the court's length, nor
    which end is which baseline, nor whether the view is mirrored.  All eight
    combinations (4 rotations x 2 handedness) are plausible a priori; the
    painted-line alignment cost picks the right one.
    """
    base = [(0.0, 0.0), (0.0, spec.width), (spec.length, spec.width), (spec.length, 0.0)]
    seqs = []
    for seq in (base, list(reversed(base))):
        for k in range(4):
            seqs.append((k, seq[k:] + seq[:k]))
    out = []
    for idx, (k, court_pts) in enumerate(seqs):
        out.append((idx, homography_dlt(quad, court_pts), court_pts))
    return out


# --------------------------------------------------------------------------- #
# Auto calibration from painted lines
# --------------------------------------------------------------------------- #
@dataclass
class AutoCalibration:
    corners: List[Point]
    homography: np.ndarray
    confidence: float
    method: str
    debug: Dict[str, Any]


class AutoCalibrator:
    """Detect the court quadrilateral from painted lines with classical CV."""

    def __init__(self, spec: CourtSpec, white_hsv: Optional[Tuple[int, int]] = None,
                 min_line_px: int = 60):
        self.spec = spec
        self.white_s_range = white_hsv or (90, 255)   # S in [0,90]: near-white
        self.min_line_px = min_line_px

    # -- masking ------------------------------------------------------------
    def line_mask(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv required")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_lo, s_hi = 0, self.white_s_range[0]
        mask = cv2.inRange(hsv, np.array([0, s_lo, 120]), np.array([180, s_hi, 255]))
        # close small gaps left by shadows; do NOT open (it erases the thin,
        # far court lines, which are exactly the ones that define the far corners)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        edges = cv2.Canny(mask, 30, 120)
        return edges

    def detect(self, frame: np.ndarray) -> Optional[AutoCalibration]:
        """Locate the court quadrilateral.

        Primary strategy: the painted lines of a court form a closed, convex
        outline, so the convex hull of the line mask (after a strong dilation to
        bridge gaps between lines) approximates the court very well.  This is far
        more stable under perspective than grouping Hough segments, whose angles
        fan out towards the vanishing point.

        Fallback: two-family Hough grouping, used when the hull is not a
        plausible quadrilateral.
        """
        if cv2 is None:  # pragma: no cover
            return None
        edges = self.line_mask(frame)
        h, w = frame.shape[:2]
        quad = self._quad_from_hull(edges, (w, h))
        method = "auto_hull"
        if quad is None:
            quad = self._quad_from_hough(edges, (w, h))
            method = "auto_hough"
        if quad is None:
            return None

        candidates: List[Optional[Sequence[Point]]] = []
        refined = refine_quad_by_fitline(edges, quad)
        if refined is not None:
            candidates.append(refined)
        candidates.append(quad)
        hough = self._quad_from_hough(edges, (w, h))
        if hough is not None:
            candidates.append(hough)
            h_ref = refine_quad_by_fitline(edges, hough)
            if h_ref is not None:
                candidates.append(h_ref)

        choice = best_calibration(candidates, edges, self.spec, (w, h))
        if choice is None:
            return None
        H, score, rot, ordered, cost = choice
        ok, why = validate_homography(H, self.spec, (w, h))
        quad_area = abs(cv2.contourArea(np.array(ordered, dtype=np.float32)))
        conf = float(np.clip(quad_area / (w * h), 0.0, 1.0))
        conf *= float(np.clip(1.0 - 12.0 * cost, 0.15, 1.0))
        return AutoCalibration(corners=[tuple(map(float, c)) for c in ordered],
                               homography=H, confidence=conf, method=method,
                               debug={"reason": why, "quad_fraction": round(conf, 3),
                                      "image_size": [w, h], "rotation": rot,
                                      "line_cost": round(float(cost), 5),
                                      "refined": refined is not None})

    # -- strategies ---------------------------------------------------------
    def _quad_from_hull(self, edges: np.ndarray, size: Tuple[int, int]
                        ) -> Optional[List[Point]]:
        """Court outline = hull of the connected white-line network.

        Painted lines touch each other at court corners, so the whole marking
        forms one large connected component.  Taking its convex hull yields the
        court quadrilateral without any dilation (which would inflate it and
        bias every line call), and picking the *largest* component ignores
        speckles, spectators' white clothing and scoreboards.
        """
        w, h = size
        num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        if num <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas) + 1
        mask = np.isin(labels, order[:6]).astype(np.uint8) * 255
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            return None
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        if cv2.contourArea(hull) < 0.05 * w * h:
            return None
        peri = cv2.arcLength(hull, True)
        for eps in (0.015, 0.025, 0.04, 0.06, 0.09):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return [tuple(map(float, p[0])) for p in approx]
        rect = cv2.minAreaRect(hull)
        if rect[1][0] < 0.15 * w or rect[1][1] < 0.15 * h:
            return None
        return [tuple(map(float, p)) for p in cv2.boxPoints(rect)]

    def _quad_from_hough(self, edges: np.ndarray, size: Tuple[int, int]
                         ) -> Optional[List[Point]]:
        w, h = size
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=self.min_line_px, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return None
        segs = [np.asarray(ln, dtype=np.float64).reshape(-1)[:4] for ln in lines]
        segs = [_canonical_direction(s) for s in segs]
        angles = np.array([np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) for s in segs])
        ang_hist, bins = np.histogram(angles % 180, bins=36, range=(0, 180))
        dominant = (bins[np.argmax(ang_hist)] + 2.5) % 180
        group_a = [s for s, a in zip(segs, angles % 180) if _ang_diff(a, dominant) < 25]
        group_b = [s for s, a in zip(segs, angles % 180)
                   if _ang_diff(a, (dominant + 90) % 180) < 25]
        if len(group_a) < 2 or len(group_b) < 2:
            return None
        top = _extreme_parallel(group_a, keep="min")
        bottom = _extreme_parallel(group_a, keep="max")
        left = _extreme_parallel(group_b, keep="min")
        right = _extreme_parallel(group_b, keep="max")
        if any(x is None for x in (top, bottom, left, right)):
            return None
        corners = []
        for a, b in ((top, left), (top, right), (bottom, right), (bottom, left)):
            p = _line_intersection(a, b)
            if p is None:
                return None
            corners.append(p)
        pts = np.array(corners, dtype=np.float32)
        if not _convex(pts):
            pts = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
        if len(pts) != 4:
            pts = cv2.boxPoints(cv2.minAreaRect(pts))
        return [tuple(map(float, p)) for p in pts]

    def calibrate_over_frames(self, frames: Sequence[np.ndarray],
                              min_confidence: float = 0.15) -> Optional[AutoCalibration]:
        """Median of per-frame corner estimates reduces occlusion noise."""
        found: List[AutoCalibration] = []
        for frame in frames:
            cal = self.detect(frame)
            if cal is not None and cal.confidence >= min_confidence:
                found.append(cal)
        if not found:
            return None
        # keep only calibrations that agree with the majority rotation
        rots = [int(c.debug.get("rotation", 0)) for c in found]
        common = max(set(rots), key=rots.count)
        found = [c for c in found if int(c.debug.get("rotation", 0)) == common] or found
        # weight by confidence when taking the median corner positions
        wts = np.array([max(1e-3, c.confidence) for c in found])
        corners_arr = np.array([c.corners for c in found])
        # keep the single best-aligned calibration instead of averaging: the
        # corner positions of a mirrored/rotated interpretation are not
        # comparable, and averaging them produces a court that fits nothing
        best_idx = int(np.argmin([c.debug.get("line_cost", 1.0) for c in found]))
        corners = found[best_idx].corners
        _ = (wts, corners_arr)  # kept for readability of the weighting above
        court_pts = [(x * self.spec.length, y * self.spec.width)
                     for x, y in CourtModel.CANONICAL_CORNERS]
        H = homography_dlt(corners, court_pts)
        return AutoCalibration(corners=[tuple(c) for c in corners], homography=H,
                               confidence=float(np.mean([c.confidence for c in found])),
                               method="auto_lines_multi",
                               debug={"n_frames_used": len(found), "rotation": common,
                                      "line_cost": float(np.mean(
                                          [c.debug.get("line_cost", 1.0) for c in found]))})


# --------------------------------------------------------------------------- #
# Small line helpers
# --------------------------------------------------------------------------- #
def _canonical_direction(line: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in line)
    if (x2, y2) < (x1, y1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return np.array([x1, y1, x2, y2])


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _line_point_side(line: np.ndarray) -> float:
    """Signed offset of a line from the frame centre (ordering proxy)."""
    x1, y1, x2, y2 = line
    nx, ny = -(y2 - y1), (x2 - x1)
    norm = np.hypot(nx, ny) or 1.0
    return float((nx * (x1 + x2) / 2 + ny * (y1 + y2) / 2) / norm)


def _extreme_parallel(lines: List[np.ndarray], keep: str) -> Optional[np.ndarray]:
    if not lines:
        return None
    offsets = np.array([_line_point_side(l) for l in lines])
    idx = int(np.argmin(offsets) if keep == "min" else np.argmax(offsets))
    return lines[idx]


def _line_intersection(a: np.ndarray, b: np.ndarray) -> Optional[Point]:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    if not np.isfinite(px) or not np.isfinite(py):
        return None
    return float(px), float(py)


def _convex(pts: np.ndarray) -> bool:
    if cv2 is None:  # pragma: no cover
        return True
    return bool(cv2.isContourConvex(pts.reshape(-1, 1, 2)))


def refine_quad_by_fitline(edges: np.ndarray, quad: Sequence[Point],
                           band_px: int = 26) -> Optional[List[Point]]:
    """Snap a coarse quad onto the painted lines by fitting each side.

    The hull of the line mask is systematically *outside* the real court because
    of the morphological closing used to bridge gaps.  Fitting a line through the
    edge pixels that lie near each side removes that bias, which directly
    improves line calls (a 10 px bias near the baseline is a 20 cm error).
    """
    if cv2 is None:  # pragma: no cover
        return None
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    ys, xs = np.nonzero(edges)
    if len(xs) < 40:
        return None
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return None
        d /= length
        n = np.array([-d[1], d[0]])
        rel = pixels - a
        along = rel @ d
        across = np.abs(rel @ n)
        # keep pixels close to the side, and inset from the corners (the corner
        # regions are shared between two sides and bias both fits)
        keep = (across < band_px) & (along > 0.08 * length) & (along < 0.92 * length)
        if keep.sum() < 15:
            return None
        sel = pixels[keep].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        lines.append((float(x0), float(y0), float(vx), float(vy)))
    refined = []
    for i in range(4):
        p = _line_intersection(
            np.array([lines[i][0], lines[i][1],
                      lines[i][0] + lines[i][2], lines[i][1] + lines[i][3]]),
            np.array([lines[(i + 1) % 4][0], lines[(i + 1) % 4][1],
                      lines[(i + 1) % 4][0] + lines[(i + 1) % 4][2],
                      lines[(i + 1) % 4][1] + lines[(i + 1) % 4][3]]))
        if p is None:
            return None
        refined.append(p)
    return refined


def best_calibration(candidates: Sequence[Sequence[Point]], edges: np.ndarray,
                     spec: CourtSpec, size: Tuple[int, int]
                     ) -> Optional[Tuple[np.ndarray, float, int, Sequence[Point]]]:
    """Pick the (quad, labeling) pair that best matches the painted lines."""
    w, h = size
    best = None
    for quad in candidates:
        if quad is None or len(quad) != 4:
            continue
        ordered = order_quad(quad)
        for k, H, court_pts in _labeling_candidates(ordered, spec):
            ok, _ = validate_homography(H, spec, (w, h))
            if not ok:
                continue
            cost = court_line_alignment_cost(CourtModel(spec, H), edges)
            # tie-break: prefer an unmirrored, unrotated interpretation
            score = cost + 0.0005 * (0 if k == 0 else 1)
            if best is None or score < best[0]:
                best = (score, H, k, ordered, cost)
    if best is None:
        return None
    return best[1], best[0], best[2], best[3], best[4]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_court(camera_cfg: dict, spec: CourtSpec,
                frame: Optional[np.ndarray] = None,
                extra_frames: Optional[Sequence[np.ndarray]] = None
                ) -> Tuple[CourtModel, Dict[str, Any]]:
    """Create a registered :class:`CourtModel` using the best available source.

    Returns ``(court, info)`` where ``info`` records how calibration happened so
    it can be written into the match JSON (reproducibility).
    """
    info: Dict[str, Any] = {"method": "none", "registered": False}
    camera_cfg = camera_cfg or {}

    H = camera_cfg.get("homography")
    if H is not None:
        court = CourtModel(spec, np.asarray(H, dtype=np.float64))
        info = {"method": "config_homography", "registered": True}
        return court, info

    corners = camera_cfg.get("court_corners_img")
    if corners:
        ordered = order_quad(corners) if len(corners) == 4 else list(corners)
        court = CourtModel.from_corners(spec, ordered)
        if frame is not None:
            ok, why = validate_homography(court.homography, spec,
                                          (frame.shape[1], frame.shape[0]))
            info = {"method": "config_corners", "registered": True,
                    "valid": ok, "note": why}
        else:
            info = {"method": "config_corners", "registered": True}
        info["corners"] = [list(map(float, c)) for c in ordered]
        return court, info

    if camera_cfg.get("auto_calibrate", True) and frame is not None:
        cal = AutoCalibrator(spec).calibrate_over_frames(
            [frame] + list(extra_frames or []))
        if cal is not None and cal.confidence > 0:
            court = CourtModel(spec, cal.homography)
            info = {"method": cal.method, "registered": True,
                    "confidence": cal.confidence,
                    "corners": [list(map(float, c)) for c in cal.corners],
                    "debug": cal.debug}
            return court, info

    # Last resort: assume the court fills the frame (correct up to a scale).
    h, w = (frame.shape[:2] if frame is not None else (1080, 1920))
    assumed = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    court = CourtModel.from_corners(spec, assumed)
    info = {"method": "assumed_full_frame", "registered": True,
            "confidence": 0.0,
            "note": "court fills the frame; in/out calls are only relative"}
    return court, info
