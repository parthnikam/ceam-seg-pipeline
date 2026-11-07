"""Stage 7: mesh simplification."""

from __future__ import annotations

import open3d as o3d


def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int,
) -> o3d.geometry.TriangleMesh:
    """Quadric-decimate a mesh down to roughly `target_triangles` triangles."""
    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
        mesh.remove_unreferenced_vertices()
        print(f"  simplified to {len(mesh.triangles):,} triangles")
    return mesh
