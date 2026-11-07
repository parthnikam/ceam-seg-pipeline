#!/usr/bin/env python3
"""Point cloud to mesh reconstruction pipeline (plan.md stages 1-8).

Stage 9 (fill holes, UVs, materials) is done in Blender on the exported
OBJ/PLY files.

  1. Load global map
  2. Remove statistical outliers
  3. Spatially partition into overlapping chunks
  4. Estimate normals (per chunk)
  5. Reconstruct surfaces (Poisson / Ball Pivoting) and merge
  6. Remove unwanted components
  7. Simplify geometry
  8. Export individual meshes (PLY / OBJ)

Chunks overlap by --overlap meters and each reconstructed chunk mesh is
trimmed back to its core box before merging, so there are no seams.

Checkpoints (intermediate .pcd/.ply) go to seg2pointcloud/checkpoints, final
meshes to seg2pointcloud/outputs.

Usage:
    python seg2pointcloud/reconstruct_mesh.py
    python seg2pointcloud/reconstruct_mesh.py --method ball --chunk-size 30
    python seg2pointcloud/reconstruct_mesh.py --resume --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from mesh_pipeline.cleaning import remove_statistical_outliers
from mesh_pipeline.components import remove_small_components, split_components
from mesh_pipeline.export import export_individual_meshes, export_mesh
from mesh_pipeline.io_utils import (
    DEFAULT_INPUT,
    checkpoint_exists,
    load_checkpoint,
    load_point_cloud,
    save_checkpoint,
)
from mesh_pipeline.normals import estimate_normals
from mesh_pipeline.partitioning import crop_mesh_to_core, partition_point_cloud
from mesh_pipeline.reconstruction import merge_meshes, reconstruct
from mesh_pipeline.simplify import simplify_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point cloud to mesh pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--method", choices=("poisson", "ball"), default="poisson")
    parser.add_argument("--depth", type=int, default=9, help="Poisson octree depth.")
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.05,
        help="Poisson density quantile below which vertices are removed (0 = off).",
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="0.05,0.1,0.2,0.5",
        help="Comma-separated ball pivoting radii.",
    )
    parser.add_argument("--nb-neighbors", type=int, default=20)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=float, default=20.0, help="Chunk cell size in meters.")
    parser.add_argument("--overlap", type=float, default=2.0, help="Chunk overlap in meters.")
    parser.add_argument("--normal-radius", type=float, default=0.1)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--orient-k", type=int, default=15)
    parser.add_argument(
        "--min-triangles", type=int, default=1000, help="Min triangles per component."
    )
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=200000,
        help="Simplification target (0 = keep unsimplified).",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse existing checkpoints where possible."
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def maybe_save(geometry, name: str, enabled: bool):
    if enabled:
        save_checkpoint(geometry, name)


def run(args: argparse.Namespace) -> o3d.geometry.TriangleMesh:
    radii = [float(r) for r in args.radii.split(",") if r.strip()]
    save = not args.no_checkpoints

    # Stages 1-2: load map and clean it
    if args.resume and checkpoint_exists("01_cleaned"):
        print("[1-2/8] Loading cleaned checkpoint")
        pcd = load_checkpoint("01_cleaned")
    else:
        print(f"[1/8] Loading point cloud: {args.input}")
        pcd = load_point_cloud(args.input)
        print(f"  {len(pcd.points):,} points")
        print("[2/8] Removing statistical outliers")
        pcd = remove_statistical_outliers(pcd, args.nb_neighbors, args.std_ratio)
        maybe_save(pcd, "01_cleaned", save)

    # Stages 3-5: partition, per-chunk normals + reconstruction, trim and merge
    if args.resume and checkpoint_exists("03_merged"):
        print("[3-5/8] Loading merged checkpoint")
        mesh = load_checkpoint("03_merged")
    else:
        print(f"[3/8] Partitioning (cells of {args.chunk_size} m, overlap {args.overlap} m)")
        chunks = partition_point_cloud(pcd, args.chunk_size, args.overlap)
        chunk_meshes = []
        for i, chunk in enumerate(chunks):
            name = f"chunks/chunk_{i:03d}"
            if args.resume and checkpoint_exists(name):
                print(f"[4-5/8] Chunk {i + 1}/{len(chunks)}: checkpoint")
                chunk_mesh = load_checkpoint(name)
            else:
                print(f"[4-5/8] Chunk {i + 1}/{len(chunks)}: {len(chunk.points.points):,} points")
                points = estimate_normals(
                    chunk.points, args.normal_radius, args.normal_max_nn, args.orient_k
                )
                chunk_mesh = reconstruct(
                    points,
                    method=args.method,
                    depth=args.depth,
                    density_quantile=args.density_quantile,
                    radii=radii,
                )
                chunk_mesh = crop_mesh_to_core(chunk_mesh, chunk.core_min, chunk.core_max)
                maybe_save(chunk_mesh, name, save)
            if len(chunk_mesh.triangles) > 0:
                chunk_meshes.append(chunk_mesh)
        print("[5/8] Merging chunk meshes")
        mesh = merge_meshes(chunk_meshes)
        maybe_save(mesh, "03_merged", save)
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Stage 6: remove unwanted components
    if args.resume and checkpoint_exists("04_filtered"):
        print("[6/8] Loading filtered checkpoint")
        mesh = load_checkpoint("04_filtered")
    else:
        print(f"[6/8] Removing components < {args.min_triangles} triangles")
        mesh = remove_small_components(mesh, args.min_triangles)
        maybe_save(mesh, "04_filtered", save)

    # Stage 7: simplify geometry
    if args.target_triangles > 0:
        if args.resume and checkpoint_exists("05_simplified"):
            print("[7/8] Loading simplified checkpoint")
            mesh = load_checkpoint("05_simplified")
        else:
            print(f"[7/8] Simplifying to ~{args.target_triangles:,} triangles")
            mesh = simplify_mesh(mesh, args.target_triangles)
            maybe_save(mesh, "05_simplified", save)
    else:
        print("[7/8] Simplification skipped")

    # Stage 8: export full mesh and individual components
    print("[8/8] Exporting meshes")
    stem = Path(args.input).stem.lower()
    paths = export_mesh(mesh, f"{stem}_mesh")
    parts = split_components(mesh)
    print(f"  {len(parts)} components")
    paths += export_individual_meshes(parts, stem=f"{stem}_part")
    for path in paths:
        print(f"  output: {path}")

    return mesh


def main() -> int:
    args = parse_args()
    try:
        mesh = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1

    if args.visualize:
        o3d.visualization.draw_geometries(
            [mesh], window_name="Reconstructed Mesh", width=1024, height=768
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
