"""One `allnew` frame: YOLO road masks → fixed-mount IPM BEV → spline.

This extract has **no depth stream**, so the ground plane is the tf_static
mount (camera height above base_footprint), not a per-frame RANSAC fit. Both
cameras' YOLO road masks are inverse-IPM'd into the left camera's BEV grid —
the right camera's grid points shifted by the stereo baseline from tf_static —
then unioned and a single track fit is run, exactly like run_yolo's pipeline
minus the depth-derived obstacle layer.
"""

from __future__ import annotations

import cv2
import numpy as np

from bev import Grid, road_blob, smooth_bev, warp_mask_to_bev
from boundaries import TrackFit, fit_track
from camera import Camera
from ground import Plane
from pipeline import FrameResult
from vis import overlay_camera, render_bev


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    if mask is None:
        return np.zeros((h, w), np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _road_component(drivable: np.ndarray) -> np.ndarray:
    """Keep the mask blob(s) connected to the road directly ahead of the cart.

    Wide campus pavement plus sidewalk folding makes the YOLO mask span the
    whole frame; stray leaks elsewhere only widen the BEV blob until its
    edge rows get dropped as FOV-clipped. Seeding at the bottom-centre keeps
    the pavement under the bumper, which is what the map should stamp.
    """
    h, w = drivable.shape[:2]
    m = (drivable > 0).astype(np.uint8)
    if not np.any(m):
        return m
    n, labels = cv2.connectedComponents(m)
    if n <= 2:
        return m
    seeds = np.unique(labels[int(0.92 * h) : , int(0.35 * w) : int(0.65 * w)])
    seeds = seeds[seeds > 0]
    if seeds.size == 0:
        return m
    return np.isin(labels, seeds).astype(np.uint8)


def _mask_bev(
    mask: np.ndarray | None,
    camera: Camera,
    plane: Plane,
    grid: Grid,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Mask → (BEV occupancy, image-space drivable mask). No obstacles (no depth)."""
    drivable = _normalize_mask(mask, (camera.height, camera.width))
    bev0 = np.zeros((grid.nx, grid.ny), np.uint8)
    if not np.any(drivable):
        return bev0, drivable
    drivable = _road_component(drivable) * 255
    if not np.any(drivable):
        return bev0, drivable
    bev = road_blob(
        smooth_bev(
            warp_mask_to_bev(drivable, camera, plane, grid, x_offset=offset[0], y_offset=offset[1])
        ),
        grid,
    )
    return bev, drivable


def process_frame_allnew(
    frame,
    camera_left: Camera,
    camera_right: Camera | None,
    grid: Grid,
    plane: Plane,
    prev_fit: TrackFit | None = None,
    baseline: tuple[float, float] = (0.0, 0.0),
) -> FrameResult:
    bgr_left = frame.bgr_left
    obstacle = np.zeros(bgr_left.shape[:2], np.uint8)

    drive_bev_l, drivable_l = _mask_bev(frame.road_left, camera_left, plane, grid)

    drive_bev_r = np.zeros((grid.nx, grid.ny), np.uint8)
    if camera_right is not None and frame.bgr_right is not None and frame.road_right is not None:
        drive_bev_r, _ = _mask_bev(frame.road_right, camera_right, plane, grid, baseline)

    drive_bev = np.bitwise_or(drive_bev_l, drive_bev_r)
    fit = fit_track(drive_bev, grid, prev=prev_fit)

    return FrameResult(
        drivable_l,
        obstacle,
        drive_bev,
        np.zeros((grid.nx, grid.ny), np.uint8),
        plane,
        fit,
        overlay_camera(bgr_left, drivable_l, obstacle, camera_left, plane, fit),
        render_bev(drive_bev, np.zeros((grid.nx, grid.ny), np.uint8), grid, fit),
    )
