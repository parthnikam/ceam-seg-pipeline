"""BEV occupancy → metric left/right edges → coupled Catmull-Rom spline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ground import Plane

if TYPE_CHECKING:
    from bev import Grid

MIN_WIDTH_M = 1.5
MAX_WIDTH_M = 10.0
LOOKAHEAD_M = 5.0
BIN_M = 0.8
FIT_XMAX_M = 10.0
NEAR_X0_M = 3.0
NEAR_X1_M = 6.0
WIDTH_TOL = 0.40


@dataclass
class TrackFit:
    x: np.ndarray
    y_left: np.ndarray
    y_right: np.ndarray
    y_center: np.ndarray
    e_y: float
    e_psi: float
    width: float
    kappa: float
    w_left: float
    w_right: float


def _catmull_rom(pts: np.ndarray, n_per: int = 6) -> np.ndarray:
    if len(pts) < 2:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    ext = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0.0, 1.0, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(pts[-1])
    return np.asarray(out, dtype=np.float64)


def _resample_xy(x: np.ndarray, y: np.ndarray, x_out: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x, y = x[order], y[order]
    uniq, idx = np.unique(x, return_index=True)
    return np.interp(x_out, uniq, y[idx], left=np.nan, right=np.nan)


def _smooth(y: np.ndarray, taps: int = 5) -> np.ndarray:
    if len(y) < taps:
        return y
    k = np.ones(taps, dtype=np.float64) / taps
    pad = taps // 2
    ext = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ext, k, mode="valid")


def _kappa_at(x: np.ndarray, y: np.ndarray, x_q: float) -> float:
    """κ = y'' / (1+y'²)^{3/2} from a local quadratic y(x)."""
    if len(x) < 5:
        return 0.0
    win = 3.0
    sel = np.abs(x - x_q) <= win
    if sel.sum() < 5:
        sel = np.ones(len(x), dtype=bool)
    try:
        a, b, _c = np.polyfit(x[sel], y[sel], 2)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    yp = 2.0 * a * x_q + b
    ypp = 2.0 * a
    return float(ypp / max((1.0 + yp * yp) ** 1.5, 1e-9))


def _edge_samples(drive_bev: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row left/right of the front blob, ignoring FOV-clipped rows and far IPM."""
    step = max(1, int(round(BIN_M / grid.cell)))
    i_max = min(grid.nx, int(round((FIT_XMAX_M - grid.xmin) / grid.cell)))
    xs, yl, yr = [], [], []
    for i in range(0, i_max, step):
        x = float(grid.x_of_row(i))
        if x > FIT_XMAX_M:
            break
        js = np.flatnonzero(drive_bev[i] > 0)
        if js.size < 3:
            continue
        # Occupancy that touches the BEV side is the camera FOV, not a road edge.
        if int(js.min()) <= 1 or int(js.max()) >= grid.ny - 2:
            continue
        width = float(grid.y_of_col(int(js.min())) - grid.y_of_col(int(js.max())))
        if width < MIN_WIDTH_M or width > MAX_WIDTH_M:
            continue
        xs.append(x)
        yl.append(float(grid.y_of_col(int(js.min()))))
        yr.append(float(grid.y_of_col(int(js.max()))))
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.asarray(xs), np.asarray(yl), np.asarray(yr)


def _keep_mask_width(x: np.ndarray, yl: np.ndarray, yr: np.ndarray) -> np.ndarray:
    """Keep rows whose width matches the near-field mask, not the far FOV fan."""
    width = yl - yr
    near = (x >= NEAR_X0_M) & (x <= NEAR_X1_M)
    if near.sum() >= 2:
        w0 = float(np.median(width[near]))
    else:
        w0 = float(np.median(width))
    tol = max(0.5, WIDTH_TOL * w0)
    return np.abs(width - w0) <= tol


def fit_track(
    drive_bev: np.ndarray,
    grid: Grid,
    prev: TrackFit | None = None,
) -> TrackFit | None:
    """Left/right from the near IPM blob. Width follows the mask, not the horizon."""
    xs, yl0, yr0 = _edge_samples(drive_bev, grid)
    if xs.size < 4:
        return None
    keep = _keep_mask_width(xs, yl0, yr0)
    xs, yl0, yr0 = xs[keep], yl0[keep], yr0[keep]
    if xs.size < 4:
        return None

    left_pts = _catmull_rom(np.stack([xs, yl0], 1))
    right_pts = _catmull_rom(np.stack([xs, yr0], 1))
    x_lo = max(float(left_pts[0, 0]), float(right_pts[0, 0]))
    x_hi = min(float(left_pts[-1, 0]), float(right_pts[-1, 0]), FIT_XMAX_M)
    if x_hi - x_lo < 2.5:
        return None
    x_out = np.linspace(x_lo, x_hi, 28)
    yl = _smooth(_resample_xy(left_pts[:, 0], left_pts[:, 1], x_out))
    yr = _smooth(_resample_xy(right_pts[:, 0], right_pts[:, 1], x_out))
    good = np.isfinite(yl) & np.isfinite(yr) & ((yl - yr) > MIN_WIDTH_M) & ((yl - yr) < MAX_WIDTH_M)
    if good.sum() < 8:
        return None
    x_out, yl, yr = x_out[good], yl[good], yr[good]
    keep = _keep_mask_width(x_out, yl, yr)
    if keep.sum() < 8:
        return None
    x_out, yl, yr = x_out[keep], yl[keep], yr[keep]
    yc = 0.5 * (yl + yr)

    if prev is not None and len(prev.x) >= 4:
        prev_w = float(np.median(prev.y_left - prev.y_right))
        cur_w = float(np.median(yl - yr))
        if abs(cur_w - prev_w) <= max(0.6, 0.35 * cur_w):
            yl = 0.78 * yl + 0.22 * np.interp(x_out, prev.x, prev.y_left)
            yr = 0.78 * yr + 0.22 * np.interp(x_out, prev.x, prev.y_right)
            yc = 0.5 * (yl + yr)

    xq = float(np.clip(LOOKAHEAD_M, x_out[0], x_out[-1]))
    y_c = float(np.interp(xq, x_out, yc))
    y_l = float(np.interp(xq, x_out, yl))
    y_r = float(np.interp(xq, x_out, yr))
    width = y_l - y_r
    if width < MIN_WIDTH_M or width > MAX_WIDTH_M:
        return None
    x1 = float(np.clip(xq + 1.2, x_out[0], x_out[-1]))
    x0 = float(np.clip(xq - 1.2, x_out[0], x_out[-1]))
    e_psi = float(np.arctan2(np.interp(x1, x_out, yc) - np.interp(x0, x_out, yc), x1 - x0 + 1e-6))
    kappa = _kappa_at(x_out, yc, xq)
    if abs(kappa) > 1.25:
        return None
    return TrackFit(
        x_out, yl, yr, yc,
        e_y=y_c,
        e_psi=e_psi,
        width=width,
        kappa=kappa,
        w_left=y_l,
        w_right=-y_r,
    )


def link_polylines(fit: TrackFit, plane: Plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nx3 camera_link points on the plane for left, right, center."""
    n = plane.normal
    d = plane.offset
    nz = n[2] if abs(n[2]) > 1e-4 else 1e-4

    def on_plane(x, y):
        z = -(n[0] * x + n[1] * y + d) / nz
        return np.stack([x, y, z], axis=1)

    return on_plane(fit.x, fit.y_left), on_plane(fit.x, fit.y_right), on_plane(fit.x, fit.y_center)
