"""Stage 4: normal estimation."""

from __future__ import annotations

import open3d as o3d


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.1,
    max_nn: int = 30,
    orient_k: int = 15,
) -> o3d.geometry.PointCloud:
    """Estimate normals and orient them consistently (tangent plane propagation)."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if orient_k and orient_k > 0:
        pcd.orient_normals_consistent_tangent_plane(orient_k)
    return pcd
