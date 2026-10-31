"""One extract frame: left+right plane, drivable, spline, ego BEV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bev import Grid, build_bev
from boundaries import TrackFit, fit_track
from camera import Camera
from ground import Plane, ransac_plane, sample_points
from segment import fuse_drivable, fuse_drivable_hybrid, fuse_yolo
from vis import overlay_camera, render_bev


@dataclass
class FrameResult:
    drivable: np.ndarray        # left-camera image-space drivable mask (for overlay)
    obstacle: np.ndarray        # left-camera image-space obstacle mask
    drive_bev: np.ndarray       # union of left + right BEV drivable occupancy
    obst_bev: np.ndarray        # union of left + right BEV obstacle occupancy
    plane: Plane | None
    fit: TrackFit | None
    overlay: np.ndarray
    bev_vis: np.ndarray


def _empty_result(bgr: np.ndarray, grid: Grid) -> FrameResult:
    empty = np.zeros(bgr.shape[:2], np.uint8)
    bev0 = np.zeros((grid.nx, grid.ny), np.uint8)
    return FrameResult(
        empty, empty, bev0, bev0, None, None,
        bgr.copy(), render_bev(bev0, bev0, grid, None),
    )


def _left_drivable(
    bgr: np.ndarray,
    camera: Camera,
    depth_m: np.ndarray,
    plane: Plane,
    yolo_road: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the left-camera drivable segmentation for the given mode."""
    if mode == "yolo":
        return fuse_yolo(yolo_road, bgr.shape[:2])
    if mode == "hybrid":
        drivable, ground, obstacle = fuse_drivable_hybrid(
            bgr, camera, depth_m, plane, yolo_road
        )
        return drivable, obstacle
    # default: planar
    drivable, ground, obstacle = fuse_drivable(
        bgr, camera, depth_m, plane, yolo_road
    )
    return drivable, obstacle


def process_frame(
    frame,
    camera_left: Camera,
    camera_right: Camera | None,
    grid: Grid,
    prev_fit: TrackFit | None = None,
    prev_plane: Plane | None = None,
    mode: str = "planar",
) -> FrameResult:
    """Process one stereo frame pair.

    Left camera runs the chosen drivable segmentation (has depth). Right camera
    runs YOLO-only (no depth stream). Both BEVs are unioned into a single
    occupancy grid, then a single track fit is run on the combined mask.

    Modes:
      planar  - planar flood-fill, YOLO as appearance prior
      yolo    - YOLO road mask only on both cameras
      hybrid  - planar drivable ∪ YOLO road (YOLO primary, planar fallback; default)
    """
    bgr_left = frame.bgr_left
    bgr_right = frame.bgr_right
    depth_left = frame.depth_m_left
    yolo_left = frame.yolo_road_left
    yolo_right = frame.yolo_road_right

    # ---- Left camera ----
    pts = sample_points(camera_left, depth_left)
    plane = ransac_plane(pts) or prev_plane
    if plane is None:
        return _empty_result(bgr_left, grid)

    drivable_l, obstacle_l = _left_drivable(
        bgr_left, camera_left, depth_left, plane, yolo_left, mode
    )
    drive_bev_l, obst_bev_l = build_bev(
        drivable_l, obstacle_l, camera_left, plane, depth_left, grid
    )

    # ---- Right camera: YOLO-only into BEV using shared plane ----
    drive_bev_r = np.zeros((grid.nx, grid.ny), np.uint8)
    obst_bev_r = np.zeros((grid.nx, grid.ny), np.uint8)
    if camera_right is not None and bgr_right is not None:
        drivable_r, _obstacle_r = fuse_yolo(yolo_right, bgr_right.shape[:2])
        drive_bev_r, obst_bev_r = build_bev(
            drivable_r, _obstacle_r, camera_right, plane, None, grid
        )

    # ---- Union both cameras' BEVs ----
    drive_bev = np.bitwise_or(drive_bev_l, drive_bev_r)
    obst_bev = np.bitwise_or(obst_bev_l, obst_bev_r)

    # ---- Single track fit on combined BEV ----
    fit = fit_track(drive_bev, grid, prev=prev_fit)

    return FrameResult(
        drivable_l,
        obstacle_l,
        drive_bev,
        obst_bev,
        plane,
        fit,
        overlay_camera(bgr_left, drivable_l, obstacle_l, camera_left, plane, fit),
        render_bev(drive_bev, obst_bev, grid, fit),
    )
