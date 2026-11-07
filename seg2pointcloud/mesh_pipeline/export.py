"""Stage 8: export meshes as PLY / OBJ."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

from .io_utils import OUTPUT_DIR


def export_mesh(
    mesh: o3d.geometry.TriangleMesh,
    stem: str,
    out_dir: Path = OUTPUT_DIR,
    formats=("ply", "obj"),
) -> list[Path]:
    """Write a mesh in each requested format; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        if not o3d.io.write_triangle_mesh(str(path), mesh):
            raise RuntimeError(f"Failed to write mesh: {path}")
        paths.append(path)
    return paths


def export_individual_meshes(
    parts,
    stem: str = "part",
    out_dir: Path = OUTPUT_DIR,
    formats=("obj",),
) -> list[Path]:
    """Export each connected component as its own mesh file."""
    paths = []
    for i, (label, n_tris, part) in enumerate(parts):
        paths.extend(
            export_mesh(part, f"{stem}_{i:03d}", out_dir=out_dir, formats=formats)
        )
    return paths
