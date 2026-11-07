"""Stamp a short-range ego corridor into an odom-frame map using VIO + IMU pose."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bev import Grid, smooth_bev
from boundaries import TrackFit
from camera import yaw_from_rot
from vis import PATH, ROAD, START, _legend, draw_cart


@dataclass
class Mosaic:
    cell: float = 0.15
    extent_m: float = 400.0
    stamp_xmax: float = 8.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    start_xy: tuple[float, float] | None = None
    start_yaw: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)
    driveable: np.ndarray = field(init=False, repr=False)
    # World centre of the map window. Defaults to (0, 0) with extent_m 400,
    # which silently drops stamps outside ±200 m when the track leaves the
    # first quadrant. Size the window from the driven path instead
    # (see zed_mapper_new.run.map_window) so nothing is cropped.
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        n = int(round(self.extent_m / self.cell))
        self.n = n + (n % 2)
        self.origin = self.n // 2
        self.driveable = np.zeros((self.n, self.n), np.float32)

    def _stamp(self, bev: np.ndarray, grid: Grid, fit: TrackFit | None) -> None:
        i, j = np.nonzero(bev)
        if i.size == 0:
            return
        xe = grid.x_of_row(i)
        ye = grid.y_of_col(j)
        keep = xe <= self.stamp_xmax
        if fit is not None and len(fit.x) >= 2:
            yl = np.interp(xe, fit.x, fit.y_left, left=float(fit.y_left[0]), right=np.nan)
            yr = np.interp(xe, fit.x, fit.y_right, left=float(fit.y_right[0]), right=np.nan)
            pad = 0.12
            keep &= np.isfinite(yl) & np.isfinite(yr) & (ye <= yl + pad) & (ye >= yr - pad)
        if not np.any(keep):
            return
        xe, ye = xe[keep], ye[keep]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xw = self.x + c * xe - s * ye
        yw = self.y + s * xe + c * ye
        row = np.rint(self.origin - (xw - self.center_x) / self.cell).astype(np.int32)
        col = np.rint(self.origin - (yw - self.center_y) / self.cell).astype(np.int32)
        on = (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)
        self.driveable[row[on], col[on]] = np.maximum(self.driveable[row[on], col[on]], 1.0)

    def add(
        self,
        T_odom_cam: np.ndarray,
        drive_bev: np.ndarray,
        obst_bev: np.ndarray,
        grid: Grid,
        fit: TrackFit | None = None,
    ) -> None:
        del obst_bev
        self.x = float(T_odom_cam[0, 3])
        self.y = float(T_odom_cam[1, 3])
        self.yaw = yaw_from_rot(T_odom_cam[:3, :3])
        if self.start_xy is None:
            self.start_xy = (self.x, self.y)
            self.start_yaw = self.yaw
        self.path.append((self.x, self.y))
        self._stamp(drive_bev, grid, fit)

    def _to_px(self, x: float, y: float, r0: int, c0: int, scale: int) -> tuple[int, int]:
        row = int((self.origin - (x - self.center_x) / self.cell - r0) * scale)
        col = int((self.origin - (y - self.center_y) / self.cell - c0) * scale)
        return col, row

    def _content_cells(self, road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        rr, cc = np.nonzero(road)
        if rr.size:
            rows.append(rr)
            cols.append(cc)
        points = list(self.path)
        if self.start_xy is not None:
            points.append(self.start_xy)
        points.append((self.x, self.y))
        for x, y in points:
            rows.append(np.array([int(round(self.origin - (x - self.center_x) / self.cell))], dtype=np.int32))
            cols.append(np.array([int(round(self.origin - (y - self.center_y) / self.cell))], dtype=np.int32))
        return np.concatenate(rows), np.concatenate(cols)

    def render(self, min_px: int = 1400, px_per_cell: int = 6) -> np.ndarray:
        """Full map, centered on a square canvas large enough to hold it."""
        road = (self.driveable > 0).astype(np.uint8) * 255
        road = smooth_bev(road, cells=2)
        if not np.any(road) and not self.path:
            empty = np.full((min_px, min_px, 3), 28, np.uint8)
            cv2.putText(empty, "no map yet", (40, min_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
            return empty
        r_all, c_all = self._content_cells(road)
        r0, r1 = int(r_all.min()), int(r_all.max())
        c0, c1 = int(c_all.min()), int(c_all.max())
        h_c, w_c = r1 - r0 + 1, c1 - c0 + 1
        side = max(h_c, w_c) + 40
        r0s = int(np.floor((r0 + r1) / 2.0 - side / 2.0))
        c0s = int(np.floor((c0 + c1) / 2.0 - side / 2.0))
        crop = np.zeros((side, side), np.uint8)
        src_r0, src_c0 = max(0, r0s), max(0, c0s)
        src_r1, src_c1 = min(self.n, r0s + side), min(self.n, c0s + side)
        dst_r0, dst_c0 = src_r0 - r0s, src_c0 - c0s
        crop[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = (
            road[src_r0:src_r1, src_c0:src_c1]
        )

        scale = max(px_per_cell, int(np.ceil(min_px / max(side, 1))))
        vis = np.full((side, side, 3), 28, np.uint8)
        vis[crop > 0] = ROAD
        vis = cv2.resize(vis, (side * scale, side * scale), interpolation=cv2.INTER_NEAREST)

        if len(self.path) >= 2:
            pts = np.int32([self._to_px(x, y, r0s, c0s, scale) for x, y in self.path])
            cv2.polylines(vis, [pts], False, PATH, max(2, scale // 3), cv2.LINE_AA)
        moved = False
        if self.start_xy is not None:
            moved = float(np.hypot(self.x - self.start_xy[0], self.y - self.start_xy[1])) > 1.5
            sx, sy = self._to_px(*self.start_xy, r0s, c0s, scale)
            if moved:
                draw_cart(vis, sx, sy, self.start_yaw, 16, "START", START, label_side=-1)
        nx, ny = self._to_px(self.x, self.y, r0s, c0s, scale)
        now_label = "NOW" if moved else "START/NOW"
        draw_cart(vis, nx, ny, self.yaw, 20, now_label)

        metres = side * self.cell
        bar_m = 10.0 if metres > 25 else 5.0
        bar_px = int(bar_m / self.cell * scale)
        bx, by = 24, vis.shape[0] - 36
        cv2.line(vis, (bx, by), (bx + bar_px, by), (220, 220, 220), 4)
        cv2.putText(vis, f"{bar_m:.0f} m", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
        _legend(vis, [("NOW", (0, 210, 255)), ("START", START), ("path", PATH), ("road", ROAD)])
        return vis
