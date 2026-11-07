"""Stage 6: connected-component filtering and splitting."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def split_components(mesh) -> list[tuple[int, int, o3d.geometry.TriangleMesh]]:
    """Split a mesh into connected components, largest first.

    Returns [(cluster_label, n_triangles, mesh), ...].
    """
    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    n_tris = np.asarray(n_tris)
    parts = []
    for label in np.argsort(n_tris)[::-1]:
        if n_tris[label] == 0:
            continue
        part = o3d.geometry.TriangleMesh(mesh)
        part.remove_triangles_by_mask(clusters != label)
        part.remove_unreferenced_vertices()
        parts.append((int(label), int(n_tris[label]), part))
    return parts


def remove_small_components(
    mesh: o3d.geometry.TriangleMesh,
    min_triangles: int = 1000,
) -> o3d.geometry.TriangleMesh:
    """Remove connected components smaller than `min_triangles` triangles."""
    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    n_tris = np.asarray(n_tris)
    small = n_tris[clusters] < min_triangles
    n_removed = int(small.sum())
    if n_removed > 0:
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
    print(
        f"  removed {n_removed:,} triangles from small components, "
        f"{len(mesh.triangles):,} triangles kept"
    )
    return mesh
