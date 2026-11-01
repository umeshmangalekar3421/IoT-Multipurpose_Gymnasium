"""Projective geometry, curve utilities and ball-trajectory maths.

Pure NumPy (no OpenCV) so the mobile / MCU-side rule engine can use it too.
Only ``cv2`` is used optionally by the court *detector*, never here.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Homography
# --------------------------------------------------------------------------- #
def _normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid at origin, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2], T


def homography_dlt(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """Least-squares homography mapping ``src`` (image) -> ``dst`` (court metres).

    Requires >= 4 non-collinear correspondences.  Works with the 4 court
    corners (fast calibration) or many clicked line intersections (refinement).
    """
    src_a = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst_a = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src_a.shape != dst_a.shape or len(src_a) < 4:
        raise ValueError("need >= 4 paired correspondences")

    src_n, T_src = _normalise_points(src_a)
    dst_n, T_dst = _normalise_points(dst_a)

    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=np.float64)

    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ Hn @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts: Iterable[Point]) -> np.ndarray:
    """Apply a 3x3 homography to an (N,2) array of points. Returns (N,2)."""
    arr = np.asarray(list(pts) if not isinstance(pts, np.ndarray) else pts,
                     dtype=np.float64).reshape(-1, 2)
    homo = np.hstack([arr, np.ones((len(arr), 1))])
    proj = homo @ H.T
    denom = proj[:, 2:3]
    # Points at / beyond the horizon produce garbage; keep them finite.
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return proj[:, :2] / denom


def apply_homography_single(H: np.ndarray, pt: Point) -> Optional[Point]:
    out = apply_homography(H, [pt])[0]
    if not np.all(np.isfinite(out)):
        return None
    return float(out[0]), float(out[1])


def rectify_court_size(sport_length: float, sport_width: float,
                       target_long_px: int = 1280) -> Tuple[int, int, float]:
    """Pixel size + scale (px per metre) for a top-down court render."""
    scale = target_long_px / float(sport_length)
    return int(round(sport_length * scale)), int(round(sport_width * scale)), scale


# --------------------------------------------------------------------------- #
# Polygons / point tests
# --------------------------------------------------------------------------- #
def polygon_mask(points: Sequence[Point], test: Sequence[Point]) -> np.ndarray:
    """Vectorised even-odd ray casting. Returns boolean (len(test),)."""
    poly = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(test, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi
        inside ^= cond & (x < x_int)
        j = i
    return inside


def point_in_polygon(points: Sequence[Point], pt: Point) -> bool:
    return bool(polygon_mask(points, [pt])[0])


def distance_to_segments(pt: Point, segments: Sequence[Tuple[Point, Point]]) -> float:
    """Minimum perpendicular distance from ``pt`` to a set of line segments."""
    if not segments:
        return float("inf")
    p = np.asarray(pt, dtype=np.float64)
    best = float("inf")
    for a, b in segments:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        d = float(np.linalg.norm(p - (a + t * ab)))
        best = min(best, d)
    return best


# --------------------------------------------------------------------------- #
# Signal utilities for trajectories
# --------------------------------------------------------------------------- #
def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average that preserves length and edges."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1 or len(v) < 3:
        return v.copy()
    window = min(window, len(v) if len(v) % 2 else len(v) - 1)
    if window < 3:
        return v.copy()
    pad = window // 2
    padded = np.pad(v, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(v)]


def savgol_smooth(values: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoother (local polynomial least squares), edge-safe."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(v)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 else n - 1)
    if window <= polyorder + 1 or n < window:
        return moving_average(v, 3)
    half = window // 2
    padded = np.pad(v, half, mode="edge")
    out = np.empty(n)
    xs = np.arange(-half, half + 1, dtype=np.float64)
    # Vandermonde pseudo-inverse is identical for every (centred) window.
    V = np.vander(xs, polyorder + 1, increasing=True)
    Vpinv = np.linalg.pinv(V)  # (p+1, window)
    coef_weights = Vpinv[0]    # evaluate polynomial at x=0
    for i in range(n):
        out[i] = float(coef_weights @ padded[i: i + window])
    return out


def local_maxima(values: np.ndarray, min_prominence: float = 0.0,
                 min_distance: int = 1) -> List[int]:
    """Indices of local maxima with prominence and minimum spacing.

    Used for bounce detection: the ball reaches its lowest image point at a
    bounce, i.e. a maximum of the *image y* signal.  Flat tops (a ball at rest on
    the floor) are collapsed to a single peak at the middle of the plateau —
    without that, a landing at the very end of a flight is invisible.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(v)
    if n < 3:
        return []

    peak_indices: List[int] = []
    i = 1
    while i < n - 1:
        if v[i] > v[i - 1]:
            j = i
            while j < n - 1 and v[j + 1] == v[i]:
                j += 1
            if j < n - 1 and v[j] > v[j + 1]:
                peak_indices.append((i + j) // 2)
            i = j + 1
        else:
            i += 1

    candidates: List[Tuple[float, int]] = []
    for idx in peak_indices:
        li = idx
        while li > 0 and v[li - 1] <= v[li]:
            li -= 1
        ri = idx
        while ri < n - 1 and v[ri + 1] <= v[ri]:
            ri += 1
        prominence = v[idx] - max(v[li], v[ri])
        if prominence >= min_prominence:
            candidates.append((float(prominence), idx))

    candidates.sort(key=lambda c: -c[0])
    chosen: List[int] = []
    for _, idx in candidates:
        if all(abs(idx - c) >= min_distance for c in chosen):
            chosen.append(idx)
    return sorted(chosen)


def parabolic_vertex(y_minus: float, y_center: float, y_plus: float) -> Tuple[float, float]:
    """Sub-sample offset + value of a parabola through 3 equally spaced samples."""
    denom = (y_minus - 2 * y_center + y_plus)
    if abs(denom) < 1e-12:
        return 0.0, float(y_center)
    offset = 0.5 * (y_minus - y_plus) / denom
    value = y_center - 0.25 * (y_minus - y_plus) * offset
    return float(offset), float(value)


def velocity(pts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-sample velocity via central differences (same length as input)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(pts) < 2:
        return np.zeros_like(pts)
    dt = np.gradient(times)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    return np.stack([np.gradient(pts[:, 0]) / dt, np.gradient(pts[:, 1]) / dt], axis=1)


def speed(pts: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity(pts, times), axis=1)


# --------------------------------------------------------------------------- #
# 3D-ish reconstruction helpers
# --------------------------------------------------------------------------- #
def fit_parabola_3d(track_xy: np.ndarray, times: np.ndarray) -> Optional[dict]:
    """Fit a projectile parabola in *court* metres assuming the ball left the floor.

    Without stereo/depth we cannot recover height, but for ball sports the
    landing point is well approximated by the vertex of a parabola fitted to the
    descending branch in court coordinates (x(t), y(t) ~ linear/quadratic).
    """
    pts = np.asarray(track_xy, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(pts) < 4:
        return None
    t0 = t - t.mean()
    coeffs = {}
    for axis, name in ((0, "x"), (1, "y")):
        A = np.stack([t0 ** 2, t0, np.ones_like(t0)], axis=1)
        sol, *_ = np.linalg.lstsq(A, pts[:, axis], rcond=None)
        coeffs[name] = sol
    cx, cy = coeffs["x"], coeffs["y"]
    residual = 0.0
    for axis, name in ((0, "x"), (1, "y")):
        c = coeffs[name]
        pred = c[0] * t0 ** 2 + c[1] * t0 + c[2]
        residual += float(np.mean((pred - pts[:, axis]) ** 2))
    return {
        "coeffs": coeffs,
        "rmse": float(np.sqrt(residual / 2.0)),
        "t_mean": float(t.mean()),
    }
