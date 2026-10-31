"""Find the connected flat surface in front of the cart, then keep the road.

A wall has *uniform image depth* (same z everywhere). A road does not: depth
grows toward the vanishing point, but 3D height above the fitted plane stays
near zero. We seed just above the hood and flood through that planar region,
then keep the pavement-looking part (YOLO / asphalt). Grass that is also
planar is dropped. Far pixels with no stereo stay only if they touch that blob.
"""

from __future__ import annotations

import cv2
import numpy as np

from camera import Camera
from ground import Plane, height_maps

GROUND_ABS_M = 0.14
OBSTACLE_M = 0.28
WALL_DEPTH_STD_M = 0.18


def hood_mask(h: int, w: int) -> np.ndarray:
    """Kart body in the bottom of the ZED X frame (test4/test5)."""
    mask = np.zeros((h, w), np.uint8)
    poly = np.int32(
        [
            [0, int(0.88 * h)],
            [int(0.10 * w), int(0.82 * h)],
            [int(0.38 * w), int(0.78 * h)],
            [int(0.70 * w), int(0.78 * h)],
            [w - 1, int(0.88 * h)],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    )
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def asphalt_prior(bgr: np.ndarray) -> np.ndarray:
    """Gray pavement, not green vegetation."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    gray = (s < 55) & (l > 35) & (l < 210)
    green = (h > 32) & (h < 95) & (s > 28) & (l > 30)
    out = np.zeros(bgr.shape[:2], np.uint8)
    out[gray & ~green] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, k)


def appearance_mask(bgr: np.ndarray, yolo_road: np.ndarray | None) -> np.ndarray:
    asphalt = asphalt_prior(bgr)
    if yolo_road is None:
        return asphalt
    h, w = bgr.shape[:2]
    near = np.zeros((h, w), np.uint8)
    near[int(0.52 * h) :, :] = 255
    # YOLO often misses pavement next to the hood; fill with asphalt there.
    return cv2.bitwise_or(yolo_road, cv2.bitwise_and(asphalt, near))


def _depth_std(depth_m: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Local depth stdev. NaNs ignored via a validity-weighted blur."""
    valid = np.isfinite(depth_m).astype(np.float32)
    z = np.nan_to_num(depth_m, nan=0.0).astype(np.float32)
    k = (ksize, ksize)
    w = cv2.blur(valid, k)
    m1 = cv2.blur(z, k) / np.maximum(w, 1e-3)
    m2 = cv2.blur(z * z, k) / np.maximum(w, 1e-3)
    var = np.maximum(m2 - m1 * m1, 0.0)
    std = np.sqrt(var)
    std[w < 0.25] = np.nan
    return std


def _keep_seeded_components(binary: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents((binary > 0).astype(np.uint8))
    if n <= 1:
        return np.zeros_like(binary)
    keep = np.unique(labels[seeds > 0])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(binary)
    out = np.zeros_like(binary)
    out[np.isin(labels, keep)] = 255
    return out


def fuse_drivable(
    bgr: np.ndarray,
    camera: Camera,
    depth_m: np.ndarray,
    plane: Plane,
    yolo_road: np.ndarray | None,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (drivable, planar-ground, obstacle) at full resolution."""
    h, w = bgr.shape[:2]
    appear = appearance_mask(bgr, yolo_road)
    height, _xyz, ok = height_maps(camera, depth_m, plane, stride=stride)
    ground_s = np.zeros(height.shape, np.uint8)
    obst_s = np.zeros(height.shape, np.uint8)
    valid = ok & np.isfinite(height)
    ground_s[valid & (np.abs(height) < GROUND_ABS_M)] = 255
    obst_s[valid & (height > OBSTACLE_M)] = 255
    ground = cv2.resize(ground_s, (w, h), interpolation=cv2.INTER_NEAREST)
    obstacle = cv2.resize(obst_s, (w, h), interpolation=cv2.INTER_NEAREST)
    hood = hood_mask(h, w)

    std = _depth_std(depth_m)
    height_full = cv2.resize(
        np.nan_to_num(height, nan=99.0).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    wall = np.isfinite(std) & (std < WALL_DEPTH_STD_M) & (height_full > 0.45)
    obstacle[wall] = 255

    has_depth = np.isfinite(depth_m)
    planar_road = (ground > 0) & (appear > 0) & (obstacle == 0) & (hood == 0)
    far_road = (appear > 0) & (~has_depth) & (obstacle == 0) & (hood == 0)

    seeds = np.zeros((h, w), np.uint8)
    y0, y1 = int(0.58 * h), int(0.78 * h)
    x0, x1 = int(0.28 * w), int(0.72 * w)
    seeds[y0:y1, x0:x1] = planar_road[y0:y1, x0:x1]

    growable = np.zeros((h, w), np.uint8)
    growable[planar_road | far_road] = 255
    drivable = _keep_seeded_components(growable, seeds)
    if (drivable > 0).sum() < 2000:
        drivable = planar_road.astype(np.uint8) * 255
    drivable[hood > 0] = 0
    drivable[obstacle > 0] = 0
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, k_open)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, k_close)
    return drivable, ground, obstacle


def fuse_yolo(yolo_road: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Drivable = YOLO road mask. Empty obstacle layer (YOLO is the segmenter)."""
    h, w = shape
    if yolo_road is None:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    mask = yolo_road
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    drivable = (mask > 0).astype(np.uint8) * 255
    return drivable, np.zeros((h, w), np.uint8)


def _validate_yolo_road(
    yolo_road: np.ndarray | None,
    planar_road: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Sanity-check a YOLO road mask before unioning it with planar drivable.

    Returns a cleaned binary YOLO mask (0/255) or None if YOLO looks broken
    (empty, absurdly large, or no overlap with the planar road region).
    """
    h, w = shape
    if yolo_road is None:
        return None
    yolo = (yolo_road > 0).astype(np.uint8)
    total = int(yolo.sum())
    if total < 200:
        return None
    # Road should not cover more than half the frame.
    if total > 0.50 * h * w:
        return None
    # If we have a planar road region, YOLO should overlap it somewhat.
    if planar_road is not None and int(planar_road.sum()) > 1000:
        overlap = int(np.logical_and(yolo > 0, planar_road > 0).sum())
        if overlap < 0.10 * int(planar_road.sum()):
            return None
    # Keep only the largest connected component near the bottom-center
    # (road should be in front of the cart, not a stray wall/building blob).
    n, labels = cv2.connectedComponents(yolo)
    if n <= 1:
        return None
    # Seed region: bottom 55-85% of height, center 30-70% of width.
    seeds = np.zeros((h, w), np.uint8)
    y0, y1 = int(0.55 * h), int(0.85 * h)
    x0, x1 = int(0.30 * w), int(0.70 * w)
    seeds[y0:y1, x0:x1] = 255
    comp_in_seed = np.unique(labels[seeds > 0])
    comp_in_seed = comp_in_seed[comp_in_seed > 0]
    if comp_in_seed.size == 0:
        return None
    best = int(comp_in_seed[np.argmax([int((labels == c).sum()) for c in comp_in_seed])])
    cleaned = np.zeros((h, w), np.uint8)
    cleaned[labels == best] = 255
    return cleaned


def fuse_drivable_hybrid(
    bgr: np.ndarray,
    camera: Camera,
    depth_m: np.ndarray,
    plane: Plane,
    yolo_road: np.ndarray | None,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hybrid: planar drivable ∪ validated YOLO road mask.

    Planar flood-fill covers where YOLO is missing / mis-calibrated; YOLO
    covers where the plane fit is wrong. YOLO is sanity-checked before
    unioning so a completely-bad YOLO frame (wall/building labeled as road,
    or no road at all) falls back to planar-only.
    """
    h, w = bgr.shape[:2]
    drivable_p, ground, obstacle = fuse_drivable(bgr, camera, depth_m, plane, yolo_road, stride=stride)
    planar_road = drivable_p.copy()

    yolo_valid = _validate_yolo_road(yolo_road, planar_road, (h, w))
    if yolo_valid is not None:
        drivable_p = cv2.bitwise_or(drivable_p, yolo_valid)

    hood = hood_mask(h, w)
    drivable_p[hood > 0] = 0
    drivable_p[obstacle > 0] = 0
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    drivable_p = cv2.morphologyEx(drivable_p, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    drivable_p = cv2.morphologyEx(drivable_p, cv2.MORPH_OPEN, k_open)
    return drivable_p, ground, obstacle
