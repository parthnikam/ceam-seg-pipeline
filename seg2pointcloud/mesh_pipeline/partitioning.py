"""Stage 3: spatial partitioning into overlapping chunks.

The map is cut into a regular grid of `chunk_size` cells, but each chunk's
point selection is grown by `overlap` meters on every side so reconstruction
has supporting geometry past the cut. Reconstructed chunk meshes are later
trimmed back to the chunk core box (crop_mesh_to_core) so neighbouring chunks
merge without seams or duplicated surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class Chunk:
    index: tuple[int, int, int]
    core_min: np.ndarray
    core_max: np.ndarray
    points: o3d.geometry.PointCloud


def partition_point_cloud(
    pcd: o3d.geometry.PointCloud,
    chunk_size: float = 20.0,
    overlap: float = 2.0,
) -> list[Chunk]:
    """Split a point cloud into overlapping grid chunks."""
    pts = np.asarray(pcd.points)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    n_cells = np.maximum(1, np.ceil((pmax - pmin) / chunk_size).astype(int))

    chunks: list[Chunk] = []
    for ix in range(n_cells[0]):
        for iy in range(n_cells[1]):
            for iz in range(n_cells[2]):
                idx = np.array([ix, iy, iz])
                core_min = pmin + idx * chunk_size
                core_max = np.minimum(pmin + (idx + 1) * chunk_size, pmax)
                sel_min = core_min - overlap
                sel_max = core_max + overlap
                mask = np.all((pts >= sel_min) & (pts <= sel_max), axis=1)
                if not mask.any():
                    continue
                chunk_pcd = pcd.select_by_index(np.flatnonzero(mask))
                chunks.append(Chunk((ix, iy, iz), core_min, core_max, chunk_pcd))

    print(f"  {len(chunks)} chunks ({chunk_size} m cells, {overlap} m overlap)")
    return chunks


def crop_mesh_to_core(
    mesh: o3d.geometry.TriangleMesh,
    core_min: np.ndarray,
    core_max: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Keep only triangles whose centroid lies inside the chunk core box."""
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return mesh
    verts = np.asarray(mesh.vertices)
    centroids = verts[tris].mean(axis=1)
    keep = np.all((centroids >= core_min) & (centroids <= core_max), axis=1)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh
