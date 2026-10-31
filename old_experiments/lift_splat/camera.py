"""Approximate pinhole model for an F1 halo / T-cam.

There is no calibration in this dataset. Defaults are a 1920x1080 onboard
view: ~80 deg horizontal FOV, ~1.15 m above the ground, pitched down so the
horizon sits in the upper third of the frame. Tune --hfov / --height / --pitch
if the BEV track looks too wide, too narrow, or shifted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Camera:
    width: int
    height: int
    hfov_deg: float = 80.0
    height_m: float = 1.15
    pitch_deg: float = 12.0

    def fx(self) -> float:
        return self.width / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))

    def intrins(self) -> torch.Tensor:
        fx = self.fx()
        fy = fx
        return torch.tensor(
            [
                [fx, 0.0, self.width / 2.0],
                [0.0, fy, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float,
        )

    def rotation(self) -> torch.Tensor:
        """Camera-to-ego rotation. Ego: x forward, y left, z up. Cam: x right, y down, z forward."""
        alpha = np.deg2rad(self.pitch_deg)
        sp, cp = np.sin(alpha), np.cos(alpha)
        return torch.tensor(
            [
                [0.0, -sp, cp],
                [-1.0, 0.0, 0.0],
                [0.0, -cp, -sp],
            ],
            dtype=torch.float,
        )

    def translation(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, self.height_m], dtype=torch.float)

    def packed(self, device: torch.device):
        """B=1, N=1 tensors in the layout Lift-Splat-Shoot expects."""
        eye = torch.eye(3, dtype=torch.float, device=device)
        zeros = torch.zeros(3, dtype=torch.float, device=device)
        return (
            self.rotation().to(device)[None, None],
            self.translation().to(device)[None, None],
            self.intrins().to(device)[None, None],
            eye[None, None],
            zeros[None, None],
        )

    def optical_axis_ground_m(self) -> float | None:
        """How far ahead the principal ray hits z=0. None if it never does."""
        pitch = np.deg2rad(self.pitch_deg)
        if abs(np.sin(pitch)) < 1e-6:
            return None
        return float(self.height_m / np.tan(pitch))


