"""Metric bird's-eye grid, image warp, and a stitched world map."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from camera import Camera
from paint import LaneFit


@dataclass
class Grid:
    xmin: float = 4.0
    xmax: float = 45.0
    ymin: float = -14.0
    ymax: float = 14.0
    cell: float = 0.12

    def __post_init__(self):
        self.nx = int(round((self.xmax - self.xmin) / self.cell))
        self.ny = int(round((self.ymax - self.ymin) / self.cell))

    def xy_maps(self):
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.ymax - (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def x_of_row(self, i: np.ndarray) -> np.ndarray:
        return self.xmin + (i + 0.5) * self.cell

    def y_of_col(self, j: np.ndarray) -> np.ndarray:
        return self.ymax - (j + 0.5) * self.cell


def warp_to_bev(image: np.ndarray, camera: Camera, grid: Grid) -> np.ndarray:
    """Sample the image at each ground-plane cell."""
    XX, YY = grid.xy_maps()
    u, v, ok = camera.ground_to_pixels(XX.ravel(), YY.ravel())
    map_x = u.reshape(grid.nx, grid.ny).astype(np.float32)
    map_y = v.reshape(grid.nx, grid.ny).astype(np.float32)
    valid = ok.reshape(grid.nx, grid.ny)
    valid &= (map_x >= 0) & (map_x < camera.width - 1) & (map_y >= 0) & (map_y < camera.height - 1)
    interp = cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_NEAREST
    warped = cv2.remap(image, map_x, map_y, interp, borderValue=0)
    warped[~valid] = 0
    return warped


def _centerline_curve(fit: LaneFit | None, camera: Camera) -> float:
    """Ground-plane d²Y/dX² of the lane centre. Yaw rate ≈ this times forward speed."""
    if fit is None or len(fit.left_xy) < 8:
        return 0.0
    mid = 0.5 * (fit.left_xy + fit.right_xy)
    X, Y, ok = camera.pixels_to_ground(mid[:, 0], mid[:, 1])
    pts = np.stack([X[ok], Y[ok]], 1)
    if len(pts) < 8:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    try:
        a, _, _ = np.polyfit(pts[:, 0], pts[:, 1], 2)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    return 0.0 if abs(a) > 0.08 else float(2.0 * a)


def _flow_motion(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    driveable: np.ndarray,
    camera: Camera,
    y_min: float,
    y_max: float,
) -> tuple[float, float] | None:
    """Forward / left translation from optical flow on the driveable band."""
    y0, y1 = int(gray.shape[0] * y_min), int(gray.shape[0] * y_max)
    scale = 0.4
    a = cv2.resize(prev_gray[y0:y1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray[y0:y1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mask = cv2.resize(driveable[y0:y1], (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask)
    if xs.size < 80:
        return None
    pick = np.random.default_rng(0).choice(xs.size, size=min(800, xs.size), replace=False)
    xs, ys = xs[pick], ys[pick]
    u0, v0 = xs / scale, ys / scale + y0
    u1 = u0 + flow[ys, xs, 0] / scale
    v1 = v0 + flow[ys, xs, 1] / scale
    X0, Y0, ok0 = camera.pixels_to_ground(u0, v0)
    X1, Y1, ok1 = camera.pixels_to_ground(u1, v1)
    keep = ok0 & ok1 & (X0 > 5.0) & (X0 < 28.0)
    if keep.sum() < 40:
        return None
    dx = float(np.median(X0[keep] - X1[keep]))
    dy = float(np.median(Y0[keep] - Y1[keep]))
    if dx < -0.5 or dx > 8.0 or abs(dy) > 3.0:
        return None
    return dx, dy


@dataclass
class Mosaic:
    cell: float = 0.25
    extent_m: float = 500.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    last_gray: np.ndarray | None = field(default=None, repr=False)
    last_motion: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        n = int(round(self.extent_m / self.cell))
        self.n = n + (n % 2)
        self.origin = self.n // 2
        self.driveable = np.zeros((self.n, self.n), np.float32)
        self.paint = np.zeros((self.n, self.n), np.float32)

    def _stamp(self, bev: np.ndarray, grid: Grid, layer: np.ndarray) -> None:
        """Paint ego-frame BEV cells into the world map at the current pose."""
        i, j = np.nonzero(bev)
        if i.size == 0:
            return
        Xe, Ye = grid.x_of_row(i), grid.y_of_col(j)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Xw = self.x + c * Xe - s * Ye
        Yw = self.y + s * Xe + c * Ye
        row = np.rint(self.origin - Xw / self.cell).astype(np.int32)
        col = np.rint(self.origin - Yw / self.cell).astype(np.int32)
        keep = (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)
        layer[row[keep], col[keep]] = np.maximum(layer[row[keep], col[keep]], 1.0)

    def add(self, frame, driveable, driveable_bev, paint_bev, fit, camera, grid, y_min, y_max) -> None:
        """Estimate motion from the last frame, then stamp this BEV onto the map."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.last_gray is not None:
            flow = _flow_motion(self.last_gray, gray, driveable, camera, y_min, y_max)
            dx, dy = flow if flow is not None else self.last_motion
            dth = _centerline_curve(fit, camera) * dx
            if abs(dth) > np.deg2rad(12):
                dth = 0.0
            if dx > 0.02:
                self.last_motion = (dx, dy)
                self.x += dx * np.cos(self.yaw) - dy * np.sin(self.yaw)
                self.y += dx * np.sin(self.yaw) + dy * np.cos(self.yaw)
                self.yaw += dth
        self.last_gray = gray
        self._stamp(driveable_bev, grid, self.driveable)
        self._stamp(paint_bev, grid, self.paint)

    def render(self, pad: int = 24) -> np.ndarray:
        """Crop the used region: green driveable, white paint, arrow at the car."""
        used = (self.driveable > 0) | (self.paint > 0)
        if not np.any(used):
            empty = np.full((400, 400, 3), 22, np.uint8)
            cv2.putText(empty, "no map yet", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            return empty
        rows, cols = np.any(used, 1), np.any(used, 0)
        r0, r1 = int(np.flatnonzero(rows)[0]), int(np.flatnonzero(rows)[-1])
        c0, c1 = int(np.flatnonzero(cols)[0]), int(np.flatnonzero(cols)[-1])
        r0, r1 = max(0, r0 - pad), min(self.n, r1 + pad + 1)
        c0, c1 = max(0, c0 - pad), min(self.n, c1 + pad + 1)
        vis = np.full((r1 - r0, c1 - c0, 3), 22, np.uint8)
        vis[self.driveable[r0:r1, c0:c1] > 0] = (36, 150, 36)
        vis[self.paint[r0:r1, c0:c1] > 0] = (230, 230, 230)
        scale = 3 if vis.shape[0] < 900 else 2
        vis = cv2.resize(vis, (vis.shape[1] * scale, vis.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        er = int((self.origin - self.x / self.cell - r0) * scale)
        ec = int((self.origin - self.y / self.cell - c0) * scale)
        if 0 <= er < vis.shape[0] and 0 <= ec < vis.shape[1]:
            tip = (int(ec - np.sin(self.yaw) * 18), int(er - np.cos(self.yaw) * 18))
            cv2.arrowedLine(vis, (ec, er), tip, (0, 185, 118), 2, tipLength=0.35)
        label = f"{(c1 - c0) * self.cell:.0f}m x {(r1 - r0) * self.cell:.0f}m"
        cv2.putText(vis, label, (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return vis
