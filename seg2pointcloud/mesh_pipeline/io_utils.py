"""Point cloud / mesh IO and checkpoint helpers.

Intermediate checkpoints are written to seg2pointcloud/checkpoints and final
meshes to seg2pointcloud/outputs, regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

PIPELINE_DIR = Path(__file__).resolve().parent
SEG2POINTCLOUD_DIR = PIPELINE_DIR.parent
REPO_ROOT = SEG2POINTCLOUD_DIR.parent

DEFAULT_INPUT = REPO_ROOT / "public" / "gocart_extract" / "allnew" / "campus" / "SurfMap.pcd"
CHECKPOINT_DIR = SEG2POINTCLOUD_DIR / "checkpoints"
OUTPUT_DIR = SEG2POINTCLOUD_DIR / "outputs"


def load_point_cloud(path=DEFAULT_INPUT) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    return pcd


def save_checkpoint(geometry, name: str) -> Path:
    """Save a point cloud (.pcd) or mesh (.ply) checkpoint; returns its path."""
    ext = ".pcd" if isinstance(geometry, o3d.geometry.PointCloud) else ".ply"
    path = CHECKPOINT_DIR / f"{name}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(geometry, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(str(path), geometry)
    else:
        o3d.io.write_triangle_mesh(str(path), geometry)
    print(f"  checkpoint: {path}")
    return path


def load_checkpoint(name: str):
    """Load a previously saved checkpoint (.pcd or .ply)."""
    for ext in (".pcd", ".ply"):
        path = CHECKPOINT_DIR / f"{name}{ext}"
        if path.exists():
            if ext == ".pcd":
                return o3d.io.read_point_cloud(str(path))
            return o3d.io.read_triangle_mesh(str(path))
    raise FileNotFoundError(f"No checkpoint '{name}' in {CHECKPOINT_DIR}")


def checkpoint_exists(name: str) -> bool:
    return any(
        (CHECKPOINT_DIR / f"{name}{ext}").exists() for ext in (".pcd", ".ply")
    )
