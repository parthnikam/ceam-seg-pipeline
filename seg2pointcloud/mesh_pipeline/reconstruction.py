"""Stage 5: per-chunk surface reconstruction and chunk merging."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    density_quantile: float = 0.05,
) -> o3d.geometry.TriangleMesh:
    """Poisson reconstruction with low-density artifact removal."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    if density_quantile > 0:
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
        mesh.remove_unreferenced_vertices()
    return mesh


def ball_pivot_mesh(
    pcd: o3d.geometry.PointCloud,
    radii=(0.05, 0.1, 0.2, 0.5),
) -> o3d.geometry.TriangleMesh:
    """Ball pivoting reconstruction for the given radius sequence."""
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([float(r) for r in radii])
    )


def reconstruct(
    pcd: o3d.geometry.PointCloud,
    method: str = "poisson",
    depth: int = 9,
    density_quantile: float = 0.05,
    radii=(0.05, 0.1, 0.2, 0.5),
) -> o3d.geometry.TriangleMesh:
    if method == "ball":
        return ball_pivot_mesh(pcd, radii)
    return poisson_mesh(pcd, depth, density_quantile)


def merge_meshes(meshes) -> o3d.geometry.TriangleMesh:
    """Merge chunk meshes and clean up duplicates left behind by trimming."""
    merged = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged += mesh
    merged.remove_duplicated_vertices()
    merged.remove_duplicated_triangles()
    merged.remove_degenerate_triangles()
    merged.remove_unreferenced_vertices()
    merged.compute_vertex_normals()
    return merged
