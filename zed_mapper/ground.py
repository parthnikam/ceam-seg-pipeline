"""RANSAC ground plane in camera_link (x forward, y left, z up)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera import Camera, R_LINK_FROM_OPT, R_OPT_FROM_LINK


@dataclass
class Plane:
    normal: np.ndarray
    offset: float

    def height(self, xyz_link: np.ndarray) -> np.ndarray:
        return xyz_link @ self.normal + self.offset


def _fit_svd(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(pts) < 20:
        return None
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    d = float(-n @ c)
    return n, d


def ransac_plane(
    xyz_link: np.ndarray,
    n_iter: int = 60,
    thresh: float = 0.08,
    rng: np.random.Generator | None = None,
) -> Plane | None:
    pts = xyz_link[np.isfinite(xyz_link).all(axis=1)]
    if len(pts) < 80:
        return None
    rng = rng or np.random.default_rng(0)
    best_n = 0
    best: tuple[np.ndarray, float] | None = None
    for _ in range(n_iter):
        i = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if n[2] < 0.4:
            continue
        d = float(-n @ p0)
        inl = np.abs(pts @ n + d) < thresh
        score = int(inl.sum())
        if score > best_n:
            refit = _fit_svd(pts[inl])
            if refit is None:
                continue
            n2, d2 = refit
            if n2[2] < 0.4:
                continue
            best_n = score
            best = (n2, d2)
    if best is None:
        return None
    return Plane(best[0], best[1])


def sample_points(
    camera: Camera,
    depth_m: np.ndarray,
    stride: int = 8,
    z_min: float = 1.5,
    z_max: float = 12.0,
) -> np.ndarray:
    """Subsample valid depth pixels → camera_link points."""
    h, w = depth_m.shape
    vs = np.arange(int(camera.cy * 0.55), h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    ok = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    if not np.any(ok):
        return np.zeros((0, 3), dtype=np.float64)
    xyz_opt = camera.unproject(uu[ok], vv[ok], z[ok])
    return Camera.optical_to_link(xyz_opt)


def pixels_to_plane(
    camera: Camera,
    plane: Plane,
    u: np.ndarray,
    v: np.ndarray,
    t_min: float = 0.6,
    t_max: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray–plane hit in camera_link (X forward, Y left). Vanishing-point rays are False."""
    shape = np.broadcast(u, v).shape
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    dir_opt = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)], axis=0)
    dir_link = R_LINK_FROM_OPT @ dir_opt
    denom = plane.normal @ dir_link
    # Ray must point toward the plane (typically downward).
    hit = denom < -1e-6
    t = np.full(u.shape, np.nan)
    t[hit] = -plane.offset / denom[hit]
    ok = hit & (t > t_min) & (t < t_max) & np.isfinite(t)
    xyz = (dir_link * t).T
    X = xyz[:, 0].reshape(shape)
    Y = xyz[:, 1].reshape(shape)
    return X, Y, ok.reshape(shape)


def plane_to_pixels(
    camera: Camera,
    plane: Plane,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project plane points (X forward, Y left) into the image."""
    shape = np.broadcast(x, y).shape
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    nz = plane.normal[2] if abs(plane.normal[2]) > 1e-6 else 1e-6
    z = -(plane.normal[0] * x + plane.normal[1] * y + plane.offset) / nz
    xyz_link = np.stack([x, y, z], axis=0)
    xyz_opt = R_OPT_FROM_LINK @ xyz_link
    zc = xyz_opt[2]
    ok = zc > 0.15
    u = camera.fx * xyz_opt[0] / np.maximum(zc, 1e-6) + camera.cx
    v = camera.fy * xyz_opt[1] / np.maximum(zc, 1e-6) + camera.cy
    ok &= np.isfinite(u) & np.isfinite(v)
    return u.reshape(shape), v.reshape(shape), ok.reshape(shape)


def height_maps(
    camera: Camera,
    depth_m: np.ndarray,
    plane: Plane,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense-ish height, camera_link xyz, and a validity mask at `stride`."""
    h, w = depth_m.shape
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    ok = np.isfinite(z) & (z > 0.4) & (z < 20.0)
    xyz = np.full((vs.size, us.size, 3), np.nan, dtype=np.float64)
    height = np.full((vs.size, us.size), np.nan, dtype=np.float64)
    if np.any(ok):
        pts = Camera.optical_to_link(camera.unproject(uu[ok], vv[ok], z[ok]))
        xyz[ok] = pts
        height[ok] = plane.height(pts)
    return height, xyz, ok
