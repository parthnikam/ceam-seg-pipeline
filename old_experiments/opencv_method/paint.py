"""White paint ridges -> left/right lines -> driveable fill."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneFit:
    left_xy: np.ndarray
    right_xy: np.ndarray


def road_band(h: int, w: int, y_min: float, y_max: float) -> tuple[np.ndarray, int, int]:
    """Trapezoid over the tarmac, cockpit punched out."""
    y0, y1 = int(h * y_min), int(h * y_max)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.int32([[0.14 * w, y0], [0.86 * w, y0], [0.99 * w, y1], [0.01 * w, y1]]),
        255,
    )
    cv2.fillConvexPoly(
        mask,
        np.int32(
            [
                [0.28 * w, y0 + 0.52 * (y1 - y0)],
                [0.72 * w, y0 + 0.52 * (y1 - y0)],
                [0.90 * w, y1],
                [0.10 * w, y1],
            ]
        ),
        0,
    )
    return mask, y0, y1


def paint_ridges(bgr: np.ndarray, y_min: float, y_max: float):
    """Top-hat ridges in the road band. Returns (tophat, binary paint, y0, y1)."""
    h, w = bgr.shape[:2]
    band, y0, y1 = road_band(h, w, y_min, y_max)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    L, S = hls[:, :, 1], hls[:, :, 2]
    equalized = cv2.createCLAHE(2.0, (8, 8)).apply(L).astype(np.float32)
    tophat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    tophat[(S > 55) | (band == 0) | (L < 55)] = 0
    vals = tophat[tophat > 0]
    cut = min(max(6.0, float(np.percentile(vals, 88)) if vals.size else 6.0), 16.0)
    paint = cv2.morphologyEx((tophat >= cut).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return tophat, paint, y0, y1


def _row_peaks(sig: np.ndarray, min_sep: int, min_val: float) -> list[int]:
    """Strongest local maxima on one image row, spaced at least min_sep apart."""
    if float(sig.max()) < min_val:
        return []
    sm = cv2.GaussianBlur(sig.reshape(1, -1), (0, 0), 2.0).ravel()
    loc = (sm[1:-1] >= sm[:-2]) & (sm[1:-1] >= sm[2:]) & (sm[1:-1] >= min_val)
    idx = np.flatnonzero(loc) + 1
    if idx.size == 0:
        return []
    picked: list[int] = []
    for i in idx[np.argsort(sm[idx])[::-1]]:
        i = int(i)
        if all(abs(i - j) >= min_sep for j in picked):
            picked.append(i)
        if len(picked) >= 8:
            break
    return sorted(picked)


def _fit_poly(ys: np.ndarray, xs: np.ndarray, deg: int, thresh: float):
    """Iteratively drop outliers and refit x = poly(y)."""
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if ys.size < 15:
        return None, None
    try:
        p = np.polyfit(ys, xs, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    keep = np.ones(ys.size, dtype=bool)
    for _ in range(3):
        keep = np.abs(np.polyval(p, ys) - xs) < thresh
        if keep.sum() < 15:
            return None, None
        try:
            p = np.polyfit(ys[keep], xs[keep], deg)
        except (np.linalg.LinAlgError, ValueError):
            return None, None
    return p, keep


def _best_poly(ys: np.ndarray, xs: np.ndarray):
    """Prefer a straight line; use a gentle quadratic only if it fits better."""
    linear = _fit_poly(ys, xs, 1, 30.0)
    quad = _fit_poly(ys, xs, 2, 22.0)
    p1, k1 = linear
    p2, k2 = quad
    if p2 is not None and abs(float(p2[0])) < 1.5e-4 and k2 is not None:
        err2 = float(np.median(np.abs(np.polyval(p2, ys[k2]) - xs[k2])))
        err1 = float(np.median(np.abs(np.polyval(p1, ys[k1]) - xs[k1]))) if p1 is not None else 1e9
        if err2 + 1.2 < err1:
            return p2, k2
    return linear


def fit_lines(tophat: np.ndarray, y0: int, y1: int, prev: LaneFit | None = None) -> LaneFit | None:
    """Outermost paint peaks per row, then a robust polyline on each side."""
    w = tophat.shape[1]
    min_sep = int(0.08 * w)
    left_x, left_y, right_x, right_y = [], [], [], []
    for v in range(y0, y1):
        sig = tophat[v]
        peak = float(sig.max())
        if peak < 5.0:
            continue
        peaks = _row_peaks(sig, min_sep, max(5.0, 0.28 * peak))
        if len(peaks) < 2:
            continue
        l, r = peaks[0], peaks[-1]
        if prev is not None and len(prev.left_xy) and len(prev.right_xy):
            pl = float(np.interp(v, prev.left_xy[:, 1], prev.left_xy[:, 0]))
            pr = float(np.interp(v, prev.right_xy[:, 1], prev.right_xy[:, 0]))
            cand_l = min(peaks, key=lambda x: abs(x - pl))
            cand_r = min(peaks, key=lambda x: abs(x - pr))
            if cand_r - cand_l >= 0.14 * w:
                l, r = cand_l, cand_r
        if r - l < 0.16 * w:
            continue
        left_x.append(l)
        left_y.append(v)
        right_x.append(r)
        right_y.append(v)

    lp, lk = _best_poly(np.array(left_y), np.array(left_x))
    rp, rk = _best_poly(np.array(right_y), np.array(right_x))
    if lp is None or rp is None or lk is None or rk is None:
        return None
    y_lo = max(float(np.array(left_y)[lk].min()), float(np.array(right_y)[rk].min()))
    y_hi = min(float(np.array(left_y)[lk].max()), float(np.array(right_y)[rk].max()))
    if y_hi - y_lo < 24:
        return None
    ys = np.arange(y_lo, y_hi + 1.0)
    lx, rx = np.polyval(lp, ys), np.polyval(rp, ys)
    if np.any(rx - lx < 40):
        return None
    return LaneFit(np.stack([lx, ys], 1), np.stack([rx, ys], 1))


def fill_driveable(fit: LaneFit | None, shape: tuple[int, int]) -> np.ndarray:
    """Polygon between the fitted left and right lines."""
    mask = np.zeros(shape, np.uint8)
    if fit is None:
        return mask
    cv2.fillPoly(mask, [np.vstack([fit.left_xy, fit.right_xy[::-1]]).astype(np.int32)], 255)
    return mask
