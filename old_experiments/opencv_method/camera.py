"""Pinhole halo cam. Ego: x forward, y left, z up. Cam: x right, y down, z forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    hfov_deg: float = 80.0
    height_m: float = 1.15
    pitch_deg: float = 16.0

    def __post_init__(self):
        self.fx = self.width / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        pitch = np.deg2rad(self.pitch_deg)
        sp, cp = np.sin(pitch), np.cos(pitch)
        self.R = np.array(
            [[0.0, -sp, cp], [-1.0, 0.0, 0.0], [0.0, -cp, -sp]],
            dtype=np.float64,
        )
        self.t = np.array([0.0, 0.0, self.height_m], dtype=np.float64)

    def pixels_to_ground(self, u: np.ndarray, v: np.ndarray):
        """Ray–plane hit of each pixel with z=0. Invalid rays are marked False."""
        shape = np.broadcast(u, v).shape
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        rays = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=0)
        dirs = self.R @ rays
        hit = dirs[2] < -1e-4
        scale = np.full(u.shape, np.nan)
        scale[hit] = -self.height_m / dirs[2, hit]
        X = (dirs[0] * scale).reshape(shape)
        Y = (dirs[1] * scale).reshape(shape)
        ok = (hit & (scale > 0.5) & np.isfinite(dirs[0] * scale)).reshape(shape)
        return X, Y, ok

    def ground_to_pixels(self, X: np.ndarray, Y: np.ndarray):
        """Project ground points (X forward, Y left) into the image."""
        shape = np.broadcast(X, Y).shape
        X = np.asarray(X, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()
        cam = self.R.T @ (np.stack([X, Y, np.zeros_like(X)]) - self.t[:, None])
        z = cam[2]
        ok = z > 0.15
        u = self.fx * cam[0] / np.maximum(z, 1e-6) + self.cx
        v = self.fy * cam[1] / np.maximum(z, 1e-6) + self.cy
        ok &= np.isfinite(u) & np.isfinite(v)
        return u.reshape(shape), v.reshape(shape), ok.reshape(shape)
