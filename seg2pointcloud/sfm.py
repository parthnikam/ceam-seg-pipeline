#!/usr/bin/env python3
"""Simple Structure from Motion using pycolmap.

Processes n frames from the left and right camera feeds of the allnew
extract, using known ZED X GS intrinsics to recover camera motion and
reconstruct 3D landmarks.

Plan reference: seg2pointcloud/plan.md
  1. Visual SLAM / Structure from Motion: recover camera movement and
     reconstruct 3D landmarks from successive images.
  2. Left/right camera intrinsics are known from zed camera info.
  3. Extrinsics are also known.

Usage:
    python seg2pointcloud/sfm.py \
        --extract public/gocart_extract/allnew \
        --output seg2pointcloud/out \
        --num-frames 50
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pycolmap

# Known ZED X GS camera intrinsics from allnew camera_info data.csv.
# Format: [fx, fy, cx, cy] for the PINHOLE model.
# Left camera:  K = [678.637, 0, 974.399; 0, 724.657, 628.182; 0, 0, 1]
# Right camera: K = [675.862, 0, 890.809; 0, 578.779, 722.269; 0, 0, 1]
CAMERAS = {
    "left": {
        "model": "PINHOLE",
        "width": 1920,
        "height": 1200,
        "params": [678.63708496, 724.65673828, 974.39916992, 628.1817627],
    },
    "right": {
        "model": "PINHOLE",
        "width": 1920,
        "height": 1200,
        "params": [675.86169434, 578.77856445, 890.80871582, 722.26867676],
    },
}


def collect_frames(extract: Path, num_frames: int) -> dict[str, list[Path]]:
    """Collect the first num_frames from left and right camera feeds."""
    left_dir = extract / "images" / "left"
    right_dir = extract / "images" / "right"

    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left image directory not found: {left_dir}")
    if not right_dir.is_dir():
        raise FileNotFoundError(f"Right image directory not found: {right_dir}")

    left_images = sorted(left_dir.glob("*.png"))[:num_frames]
    right_images = sorted(right_dir.glob("*.png"))[:num_frames]

    if not left_images:
        raise FileNotFoundError(f"No left images found in {left_dir}")
    if not right_images:
        raise FileNotFoundError(f"No right images found in {right_dir}")

    return {"left": left_images, "right": right_images}


def prepare_image_dir(frames: dict[str, list[Path]], work_dir: Path) -> Path:
    """Copy selected frames into a clean working image directory with camera prefixes."""
    img_dir = work_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    for cam_name, cam_frames in frames.items():
        for i, img_path in enumerate(cam_frames):
            dst = img_dir / f"{cam_name}_{i:06d}_{img_path.name}"
            shutil.copy2(img_path, dst)

    return img_dir


def write_known_intrinsics(work_dir: Path) -> None:
    """Write known intrinsics to a text file for reference."""
    cam_dir = work_dir / "cameras"
    cam_dir.mkdir(parents=True, exist_ok=True)
    with open(cam_dir / "known_intrinsics.txt", "w", encoding="utf-8") as f:
        f.write("# Known ZED X GS intrinsics [fx, fy, cx, cy]\n")
        for name, cam in CAMERAS.items():
            params = " ".join(f"{p:.8f}" for p in cam["params"])
            f.write(f"{name} {cam['model']} {cam['width']} {cam['height']} {params}\n")


def _open_database(db_path: Path) -> pycolmap.Database:
    """Open a COLMAP database across pycolmap versions (3.12+/4.x and older)."""
    if hasattr(pycolmap.Database, "open"):
        return pycolmap.Database.open(str(db_path))
    return pycolmap.Database(str(db_path))


def run_sfm(
    image_dir: Path,
    output_dir: Path,
    refine_intrinsics: bool = False,
) -> pycolmap.ReconstructionManager:
    """Run the pycolmap SfM pipeline with known intrinsics.

    Steps:
      1. Build a COLMAP database with known left/right camera intrinsics (and rigs/frames).
      2. Extract SIFT features.
      3. Match features exhaustively.
      4. Run incremental structure-from-motion.
      5. Write the sparse reconstruction to output_dir/sparse/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "database.db"

    if db_path.exists():
        db_path.unlink()

    # ---- Build COLMAP database with known cameras ----
    with _open_database(db_path) as db:
        with pycolmap.DatabaseTransaction(db):
            cam_ids: dict[str, int] = {}
            rig_ids: dict[str, int] = {}

            for name, cam in CAMERAS.items():
                camera = pycolmap.Camera(
                    model=cam["model"],
                    width=cam["width"],
                    height=cam["height"],
                    params=cam["params"],
                    has_prior_focal_length=True,
                )
                if hasattr(camera, "source") and hasattr(pycolmap, "CameraSource"):
                    camera.source = pycolmap.CameraSource.USER

                cam_id = db.write_camera(camera)
                camera.camera_id = cam_id
                cam_ids[name] = cam_id

                # In pycolmap >= 3.12 / 4.x, DatabaseCache requires each image to
                # belong to a Frame associated with a Rig when rigs/frames exist.
                if hasattr(db, "write_rig") and hasattr(pycolmap, "Rig"):
                    rig = pycolmap.Rig()
                    rig.add_ref_sensor(camera.sensor_id)
                    rig_ids[name] = db.write_rig(rig)

            # Add images, assigning left/right camera by filename prefix.
            for img_path in sorted(image_dir.iterdir()):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                cam_name = "left" if img_path.stem.startswith("left_") else "right"
                image = pycolmap.Image(
                    name=img_path.name,
                    camera_id=cam_ids[cam_name],
                )
                image_id = db.write_image(image)
                image.image_id = image_id

                if hasattr(db, "write_frame") and hasattr(pycolmap, "Frame"):
                    frame = pycolmap.Frame()
                    frame.rig_id = rig_ids[cam_name]
                    frame.add_data_id(image.data_id)
                    frame_id = db.write_frame(frame)
                    image.frame_id = frame_id
                    db.update_image(image)

    # ---- Feature extraction ----
    print("Extracting features...")
    pycolmap.extract_features(str(db_path), str(image_dir))

    # ---- Feature matching (exhaustive, fine for small n) ----
    print("Matching features...")
    pycolmap.match_exhaustive(str(db_path))

    # ---- Incremental reconstruction ----
    print("Running incremental reconstruction...")
    with _open_database(db_path) as db:
        cache_options = pycolmap.DatabaseCacheOptions()
        if hasattr(pycolmap.DatabaseCache, "create"):
            cache = pycolmap.DatabaseCache.create(db, cache_options)
        else:
            cache = pycolmap.DatabaseCache()
            cache.load(db, cache_options)

    manager = pycolmap.ReconstructionManager()
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = refine_intrinsics
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = refine_intrinsics
    if hasattr(options, "image_path"):
        options.image_path = str(image_dir)

    pipeline = pycolmap.IncrementalPipeline(options, cache, manager)
    pipeline.run()

    sparse_dir = output_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    manager.write(str(sparse_dir))

    return manager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple SfM with pycolmap on the allnew extract."
    )
    parser.add_argument(
        "--extract",
        type=Path,
        required=True,
        help="Path to the allnew extract folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seg2pointcloud/out"),
        help="Output directory for SfM results.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames per camera to process (default: 50).",
    )
    parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="Allow bundle adjustment to refine the known focal lengths.",
    )
    args = parser.parse_args()

    if not args.extract.is_dir():
        print(f"error: extract directory not found: {args.extract}", file=sys.stderr)
        return 1

    print(f"Collecting {args.num_frames} frames per camera from {args.extract}")
    frames = collect_frames(args.extract, args.num_frames)
    print(f"  Left:  {len(frames['left'])} frames")
    print(f"  Right: {len(frames['right'])} frames")

    print(f"Preparing image directory in {args.output / 'images'}")
    img_dir = prepare_image_dir(frames, args.output)
    print(f"  {len(list(img_dir.iterdir()))} images copied")

    write_known_intrinsics(args.output)

    print("Running SfM pipeline...")
    manager = run_sfm(
        img_dir,
        args.output,
        refine_intrinsics=args.refine_intrinsics,
    )

    print("\nSfM complete.")
    print(f"  Reconstructions:   {manager.size()}")
    for idx in range(manager.size()):
        rec = manager.get(idx)
        print(
            f"  Model {idx}:           "
            f"{rec.num_reg_images()} registered images, "
            f"{rec.num_points3D()} 3D points"
        )
    print(f"  Output:            {args.output}")
    print(f"  Sparse model:      {args.output / 'sparse'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
