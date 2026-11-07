"""Stage 2: statistical outlier removal."""

from __future__ import annotations

import open3d as o3d


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Drop points whose mean neighbour distance is a statistical outlier."""
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(
        f"  removed {len(pcd.points) - len(cleaned.points):,} outliers, "
        f"{len(cleaned.points):,} points kept"
    )
    return cleaned
