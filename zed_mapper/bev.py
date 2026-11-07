"""Ego bird's-eye occupancy.

The 2D road mask is lifted by inverse IPM: each BEV cell is a ray–plane hit,
sampled in the image. Stereo depth is only used for off-plane obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera import Camera
from ground import Plane, plane_to_pixels


@dataclass
class Grid:
    xmin: float = 2.0
    xmax: float = 12.0
    ymin: float = -6.0
    ymax: float = 6.0
    cell: float = 0.12

    def __post_init__(self):
        self.nx = int(round((self.xmax - self.xmin) / self.cell))
        self.ny = int(round((self.ymax - self.ymin) / self.cell))

    def xy_maps(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.ymax - (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def x_of_row(self, i: np.ndarray) -> np.ndarray:
        return self.xmin + (i + 0.5) * self.cell

    def y_of_col(self, j: np.ndarray) -> np.ndarray:
        return self.ymax - (j + 0.5) * self.cell


def warp_mask_to_bev(
    mask: np.ndarray,
    camera: Camera,
    plane: Plane,
    grid: Grid,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    """Inverse IPM: sample the 2D mask at each ground-plane cell.

    Image-border and upper-frame samples are dropped so FOV clips and the
    horizon never become fake left/right walls. `x_offset` / `y_offset` shift
    the grid points into another camera frame that shares this plane and the
    grid orientation (stereo baseline), so a second camera can be warped into
    the same grid.
    """
    xx, yy = grid.xy_maps()
    xx = xx + x_offset
    yy = yy + y_offset
    u, v, ok = plane_to_pixels(camera, plane, xx, yy)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    h, w = mask.shape[:2]
    margin = 16
    v_horizon = 0.42 * h
    in_img = (
        ok
        & (map_x >= margin)
        & (map_x < w - 1 - margin)
        & (map_y >= v_horizon)
        & (map_y < h - 1)
    )
    warped = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    warped[~in_img] = 0
    return warped


def rasterize(xyz_link: np.ndarray, grid: Grid) -> np.ndarray:
    occ = np.zeros((grid.nx, grid.ny), np.uint8)
    if xyz_link.size == 0:
        return occ
    i = np.rint((xyz_link[:, 0] - grid.xmin) / grid.cell - 0.5).astype(np.int32)
    j = np.rint((grid.ymax - xyz_link[:, 1]) / grid.cell - 0.5).astype(np.int32)
    keep = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
    occ[i[keep], j[keep]] = 255
    return occ


def mask_to_link_depth(
    mask: np.ndarray,
    camera: Camera,
    depth_m: np.ndarray | None,
    stride: int = 2,
) -> np.ndarray:
    """Stereo unproject — for obstacles that do not lie on the pavement plane."""
    if depth_m is None:
        return np.zeros((0, 3), dtype=np.float64)
    h, w = mask.shape
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    on = mask[vv, uu] > 0
    z = depth_m[vv, uu]
    ok = on & np.isfinite(z) & (z > 0.6) & (z < 20.0)
    if not np.any(ok):
        return np.zeros((0, 3), dtype=np.float64)
    return Camera.optical_to_link(camera.unproject(uu[ok], vv[ok], z[ok]))


def smooth_bev(bev: np.ndarray, cells: int = 3) -> np.ndarray:
    """Close holes then drop speckle. `cells` is kernel size in grid cells."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cells, cells))
    out = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, k)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return out


def road_blob(drive_bev: np.ndarray, grid: Grid) -> np.ndarray:
    """Keep the connected pavement patch that includes the cells in front of the cart."""
    occ = (drive_bev > 0).astype(np.uint8)
    if not np.any(occ):
        return occ
    n, labels = cv2.connectedComponents(occ)
    if n <= 1:
        return occ * 255
    i1 = max(1, int(round(3.5 / grid.cell)))
    j_mid = grid.ny // 2
    j_half = max(1, int(round(1.2 / grid.cell)))
    j0, j1 = max(0, j_mid - j_half), min(grid.ny, j_mid + j_half)
    seeds = np.unique(labels[:i1, j0:j1])
    seeds = seeds[seeds > 0]
    if seeds.size == 0:
        # Largest near-field component.
        near = labels[:i1]
        counts = np.bincount(near.ravel())
        counts[0] = 0
        if counts.size <= 1 or counts.max() == 0:
            return occ * 255
        seeds = np.array([int(np.argmax(counts))])
    out = np.zeros_like(occ)
    out[np.isin(labels, seeds)] = 255
    return out


def build_bev(
    drivable: np.ndarray,
    obstacle: np.ndarray,
    camera: Camera,
    plane: Plane,
    depth_m: np.ndarray,
    grid: Grid,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    drive_bev = road_blob(
        smooth_bev(warp_mask_to_bev(drivable, camera, plane, grid, x_offset, y_offset)),
        grid,
    )
    obst_xyz = mask_to_link_depth(obstacle, camera, depth_m)
    return drive_bev, rasterize(obst_xyz, grid)
