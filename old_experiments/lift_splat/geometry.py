"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License.
See LICENSE in this folder, originally at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler

Adapted from lift-splat-shoot src/models.py and src/tools.py.

Kept: BEV grid helpers, the image frustum, camera unprojection, and the
voxel-pooling splat. Dropped: EfficientNet camera encoder, BEV ResNet,
nuScenes loaders, lidar, and trajectory shooting.
"""

from __future__ import annotations

import torch


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]], dtype=torch.float)
    bx = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]],
        dtype=torch.float,
    )
    nx = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]],
        dtype=torch.long,
    )
    return dx, bx, nx


def create_frustum(final_dim, downsample, dbound):
    """Depth-discretized grid in the image plane. D x fH x fW x 3 (u, v, d)."""
    ogfH, ogfW = final_dim
    fH, fW = ogfH // downsample, ogfW // downsample
    ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape
    xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
    return torch.stack((xs, ys, ds), -1)


def get_geometry(frustum, rots, trans, intrins, post_rots, post_trans):
    """Ego-frame (x, y, z) of each frustum point. B x N x D x fH x fW x 3."""
    B, N, _ = trans.shape

    points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
    points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

    points = torch.cat(
        (
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3],
        ),
        5,
    )
    combine = rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += trans.view(B, N, 1, 1, 1, 3)
    return points


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


def voxel_pooling(geom_feats, x, bx, dx, nx):
    """Splat B x N x D x H x W x C features onto a B x C x Z x X x Y grid, then collapse Z."""
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    x = x.reshape(Nprime, C)
    geom_feats = ((geom_feats - (bx - dx / 2.0)) / dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat(
        [
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ]
    )
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    kept = (
        (geom_feats[:, 0] >= 0)
        & (geom_feats[:, 0] < nx[0])
        & (geom_feats[:, 1] >= 0)
        & (geom_feats[:, 1] < nx[1])
        & (geom_feats[:, 2] >= 0)
        & (geom_feats[:, 2] < nx[2])
    )
    x = x[kept]
    geom_feats = geom_feats[kept]

    if x.shape[0] == 0:
        return torch.zeros((B, C * int(nx[2]), int(nx[0]), int(nx[1])), device=x.device)

    ranks = (
        geom_feats[:, 0] * (nx[1] * nx[2] * B)
        + geom_feats[:, 1] * (nx[2] * B)
        + geom_feats[:, 2] * B
        + geom_feats[:, 3]
    )
    sorts = ranks.argsort()
    x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
    x, geom_feats = cumsum_trick(x, geom_feats, ranks)

    final = torch.zeros(
        (B, C, int(nx[2]), int(nx[0]), int(nx[1])),
        device=x.device,
        dtype=x.dtype,
    )
    final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
    return torch.cat(final.unbind(dim=2), 1)
