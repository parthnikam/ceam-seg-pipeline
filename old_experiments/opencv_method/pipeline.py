"""One frame: find paint, fill driveable, warp to BEV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera import Camera
from ground import Grid, warp_to_bev
from paint import LaneFit, fill_driveable, fit_lines, paint_ridges


@dataclass
class FrameResult:
    paint: np.ndarray
    paint_bev: np.ndarray
    driveable: np.ndarray
    driveable_bev: np.ndarray
    fit: LaneFit | None
    overlay: np.ndarray
    bev_vis: np.ndarray


def overlay_camera(frame: np.ndarray, paint: np.ndarray, driveable: np.ndarray, fit: LaneFit | None) -> np.ndarray:
    """Green driveable band, yellow paint, orange/blue fitted lines."""
    color = np.zeros_like(frame)
    color[driveable > 0] = (0, 190, 0)
    color[paint > 0] = (0, 255, 255)
    out = cv2.addWeighted(frame, 1.0, color, 0.32, 0)
    if fit is not None:
        cv2.polylines(out, [np.int32(fit.left_xy)], False, (0, 80, 255), 3, cv2.LINE_AA)
        cv2.polylines(out, [np.int32(fit.right_xy)], False, (255, 80, 0), 3, cv2.LINE_AA)
    return out


def render_bev(paint_bev: np.ndarray, driveable: np.ndarray, grid: Grid, scale: int = 2) -> np.ndarray:
    """Ego BEV with the car at the bottom."""
    vis = np.full((grid.nx, grid.ny, 3), 22, np.uint8)
    vis[driveable > 0] = (40, 160, 40)
    vis[paint_bev > 0] = (230, 230, 230)
    vis = np.flipud(vis)
    vis = cv2.resize(vis, (grid.ny * scale, grid.nx * scale), interpolation=cv2.INTER_NEAREST)
    cv2.circle(vis, (vis.shape[1] // 2, vis.shape[0] - 8), 5, (0, 185, 118), -1)
    return vis


def process_frame(
    frame: np.ndarray,
    camera: Camera,
    grid: Grid,
    prev_fit: LaneFit | None = None,
    y_min: float = 0.28,
    y_max: float = 0.48,
) -> FrameResult:
    """Paint lines -> fill -> ground-plane BEV for one cockpit frame."""
    h, w = frame.shape[:2]
    if camera.width != w or camera.height != h:
        camera = Camera(w, h, camera.hfov_deg, camera.height_m, camera.pitch_deg)
    tophat, paint, y0, y1 = paint_ridges(frame, y_min, y_max)
    fit = fit_lines(tophat, y0, y1, prev=prev_fit) or (
        fit_lines(tophat, y0, y1) if prev_fit is not None else None
    )
    driveable = fill_driveable(fit, paint.shape)
    paint_bev = warp_to_bev(paint, camera, grid)
    driveable_bev = warp_to_bev(driveable, camera, grid)
    return FrameResult(
        paint,
        paint_bev,
        driveable,
        driveable_bev,
        fit,
        overlay_camera(frame, paint, driveable, fit),
        render_bev(paint_bev, driveable_bev, grid),
    )
