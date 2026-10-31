"""Lift a road mask into a bird's-eye grid using LSS frustum + splat.

Camera features are the YOLO road mask, not EfficientNet. Depth is a
peaked distribution around the ground-plane hit (the right prior for
tarmac) unless depth_mode='uniform'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera
from .geometry import create_frustum, gen_dx_bx, get_geometry, voxel_pooling


@dataclass
class GridConf:
    xbound: tuple[float, float, float] = (0.0, 30.0, 0.5)
    ybound: tuple[float, float, float] = (-10.0, 10.0, 0.5)
    zbound: tuple[float, float, float] = (-10.0, 10.0, 20.0)
    dbound: tuple[float, float, float] = (2.0, 32.0, 0.5)


class LiftSplat:
    def __init__(
        self,
        camera: Camera,
        grid: GridConf | None = None,
        downsample: int = 4,
        depth_sigma: float = 1.0,
        device: str = "cpu",
    ):
        self.camera = camera
        self.grid = grid or GridConf()
        self.downsample = downsample
        self.depth_sigma = depth_sigma
        self.device = torch.device(device)

        self.dx, self.bx, self.nx = gen_dx_bx(
            self.grid.xbound, self.grid.ybound, self.grid.zbound
        )
        self.dx = self.dx.to(self.device)
        self.bx = self.bx.to(self.device)
        self.nx = self.nx.to(self.device)

        final_dim = (camera.height, camera.width)
        if camera.height % downsample or camera.width % downsample:
            raise ValueError(
                f"final_dim {(camera.height, camera.width)} must be divisible by downsample={downsample}"
            )
        self.frustum = create_frustum(final_dim, downsample, self.grid.dbound).to(self.device)
        self.D, self.fH, self.fW, _ = self.frustum.shape

    def ground_depth(self) -> torch.Tensor:
        """Camera-z of the z=0 plane at each frustum pixel. Invalid rays are 0."""
        u = self.frustum[0, :, :, 0]
        v = self.frustum[0, :, :, 1]
        ones = torch.ones_like(u)
        pix = torch.stack((u, v, ones), dim=0).reshape(3, -1)
        intrins = self.camera.intrins().to(self.device)
        rot = self.camera.rotation().to(self.device)
        height = self.camera.height_m
        ray = torch.inverse(intrins).matmul(pix)
        direction = rot.matmul(ray)
        z_dir = direction[2]
        valid = z_dir < -1e-4
        depth = torch.zeros_like(z_dir)
        depth[valid] = -height / z_dir[valid]
        d0, d1, _ = self.grid.dbound
        depth[(depth < d0) | (depth > d1)] = 0.0
        return depth.reshape(self.fH, self.fW)

    def depth_weights(self, mode: str) -> torch.Tensor:
        """D x fH x fW distribution over frustum depth bins."""
        if mode == "uniform":
            return torch.full(
                (self.D, self.fH, self.fW),
                1.0 / self.D,
                device=self.device,
            )
        if mode != "ground":
            raise ValueError(f"unknown depth mode: {mode}")

        d_ground = self.ground_depth()
        ds = self.frustum[:, 0, 0, 2]
        sigma = max(self.depth_sigma, 1e-3)
        logits = -0.5 * ((ds[:, None, None] - d_ground[None]) / sigma) ** 2
        logits = logits.masked_fill(d_ground[None] <= 0, -1e9)
        weights = torch.softmax(logits, dim=0)
        weights = weights.masked_fill(d_ground[None] <= 0, 0.0)
        return weights

    @torch.no_grad()
    def lift(
        self,
        mask: np.ndarray,
        rgb: np.ndarray | None = None,
        depth_mode: str = "ground",
        y_max: float = 0.62,
        y_min: float = 0.32,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        mask: HxW float in [0, 1], already at camera.width/height.
        rgb:  HxW x3 float in [0, 1], RGB. Colour BEV uses every ground-plane
        ray; occupancy uses `mask` only.
        Returns occupancy (nx, ny) and optional colour (nx, ny, 3).
        """
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        h = mask_t.shape[0]
        mask_t[: int(y_min * h)] = 0.0
        mask_t[int(y_max * h) :] = 0.0

        mask_s = F.adaptive_avg_pool2d(mask_t[None, None], (self.fH, self.fW))[0, 0]
        alpha = self.depth_weights(depth_mode)
        visible = (alpha.sum(0) > 0).float()
        visible[: int(y_min * self.fH)] = 0.0
        visible[int(y_max * self.fH) :] = 0.0
        occ = alpha * mask_s[None]
        weight = alpha * visible[None]
        feats = occ[..., None]

        if rgb is not None:
            rgb_t = torch.from_numpy(rgb.astype(np.float32)).to(self.device)
            if rgb_t.shape[0] != mask_t.shape[0] or rgb_t.shape[1] != mask_t.shape[1]:
                raise ValueError("rgb and mask spatial size must match")
            rgb_s = F.adaptive_avg_pool2d(
                rgb_t.permute(2, 0, 1)[None], (self.fH, self.fW)
            )[0].permute(1, 2, 0)
            colour = weight[..., None] * rgb_s[None]
            feats = torch.cat((feats, weight[..., None], colour), dim=-1)

        rots, trans, intrins, post_rots, post_trans = self.camera.packed(self.device)
        geom = get_geometry(self.frustum, rots, trans, intrins, post_rots, post_trans)
        x = feats[None, None]
        bev = voxel_pooling(geom, x, self.bx, self.dx, self.nx)[0]
        nx, ny = int(self.nx[0]), int(self.nx[1])
        occupancy = bev[0].reshape(nx, ny).detach().cpu().numpy()
        if rgb is None:
            return occupancy, None
        mass = bev[1].reshape(nx, ny).detach().cpu().numpy()
        colour_sum = bev[2:5].reshape(3, nx, ny).detach().cpu().numpy()
        denom = np.maximum(mass, 1e-6)
        colour = np.clip(colour_sum / denom, 0.0, 1.0).transpose(1, 2, 0)
        colour[mass <= 1e-4] = 0.0
        return occupancy, colour
