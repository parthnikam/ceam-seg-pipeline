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

    # Select frames by odometry displacement (>=1 m apart) to avoid
    # degenerate low-parallax bundle adjustments:
    python seg2pointcloud/sfm.py \
        --extract public/gocart_extract/allnew \
        --output seg2pointcloud/out \
        --num-frames 50 --min-translation 1.0

    # Additionally anchor camera positions to wheel odometry pose priors:
    python seg2pointcloud/sfm.py \
        --extract public/gocart_extract/allnew \
        --output seg2pointcloud/out \
        --num-frames 50 --min-translation 1.0 --use-odom-priors
"""

from __future__ import annotations

import argparse
import bisect as _bisect
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pycolmap

NUM_THREADS = max(1, os.cpu_count() or 1)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ODOM_TOLERANCE_NS = 2 * 10**9  # max gap between image time and nearest odom sample
_NO_ODOM_FALLBACK_DT_NS = 10**9  # min frame spacing when odometry is unavailable (1 s)

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


def _parse_frame_time_ns(path: Path) -> int | None:
    """Parse 'YYYYMMDD_HHMMSS_ffffff.png' filenames into UTC epoch nanoseconds."""
    try:
        dt = datetime.strptime(path.stem, "%Y%m%d_%H%M%S_%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(round((dt - _EPOCH).total_seconds() * 1e9))


def load_odometry(extract: Path):
    """Load /odom samples: (times_ns list, positions (N,3), quats xyzw (N,4)) or None."""
    odom_csv = extract / "odom" / "data.csv"
    if not odom_csv.exists():
        return None
    rows = []
    with open(odom_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                msg = json.loads(row["message_json"])
                pose = msg["pose"]["pose"]
                pos = pose["position"]
                q = pose["orientation"]
                rows.append(
                    (
                        int(row["timestamp_ns"]),
                        (pos["x"], pos["y"], pos["z"]),
                        (q["x"], q["y"], q["z"], q["w"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    times = [r[0] for r in rows]
    positions = np.array([r[1] for r in rows], dtype=np.float64)
    quats = np.array([r[2] for r in rows], dtype=np.float64)
    return times, positions, quats


def _odom_lookup(odometry, t_ns: int):
    """Return (base_link position, quaternion xyzw) nearest to t_ns, or None."""
    if odometry is None:
        return None
    times, positions, quats = odometry
    idx = _bisect.bisect_left(times, t_ns)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(times)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(times[i] - t_ns))
    if abs(times[best] - t_ns) > _ODOM_TOLERANCE_NS:
        return None
    return positions[best], quats[best]


def load_tf_static_extrinsics(extract: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load base_link -> zedxgs_{left,right}_camera_link transforms as {cam: (R, t)}."""
    tf_csv = extract / "tf_static" / "data.csv"
    link_to_cam = {"zedxgs_left_camera_link": "left", "zedxgs_right_camera_link": "right"}
    extrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not tf_csv.exists():
        return extrinsics
    with open(tf_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                msg = json.loads(row["message_json"])
            except json.JSONDecodeError:
                continue
            for tr in msg.get("transforms", []):
                if tr.get("child_frame_id") not in link_to_cam:
                    continue
                cam = link_to_cam[tr["child_frame_id"]]
                t = tr["transform"]["translation"]
                q = tr["transform"]["rotation"]
                R = _quat_xyzw_to_rotmat((q["x"], q["y"], q["z"], q["w"]))
                extrinsics[cam] = (R, np.array([t["x"], t["y"], t["z"]], dtype=np.float64))
    return extrinsics


def _quat_xyzw_to_rotmat(q) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def collect_frames(
    extract: Path,
    num_frames: int,
    min_translation: float = 1.0,
    odometry=None,
) -> dict[str, list[tuple[Path, tuple[np.ndarray, np.ndarray] | None]]]:
    """Collect frames from both cameras, gated by odometry displacement.

    Frames from left and right feeds are merged on a single timeline and a
    frame is only selected when the base_link has moved at least
    `min_translation` meters (per /odom) since the previously selected frame
    (from either camera). This keeps consecutive baselines healthy relative
    to scene depth and avoids near-duplicate left/right poses, both of which
    cause ill-conditioned bundle adjustments.

    Returns {cam: [(path, odom_sample or None), ...]}.
    """
    left_dir = extract / "images" / "left"
    right_dir = extract / "images" / "right"

    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left image directory not found: {left_dir}")
    if not right_dir.is_dir():
        raise FileNotFoundError(f"Right image directory not found: {right_dir}")

    timeline: list[tuple[int, str, Path]] = []
    for cam, cam_dir in (("left", left_dir), ("right", right_dir)):
        for path in cam_dir.glob("*.png"):
            t = _parse_frame_time_ns(path)
            if t is not None:
                timeline.append((t, cam, path))
    if not timeline:
        raise FileNotFoundError(f"No timestamped images found under {extract / 'images'}")
    timeline.sort(key=lambda item: (item[0], item[1]))

    selected: dict[str, list] = {"left": [], "right": []}
    total = 0
    last_t: int | None = None
    last_pos: np.ndarray | None = None

    for t, cam, path in timeline:
        if num_frames > 0 and total >= num_frames:
            break
        sample = _odom_lookup(odometry, t)

        if total == 0 or min_translation <= 0:
            take = True
        elif sample is not None and last_pos is not None:
            take = float(np.linalg.norm(sample[0] - last_pos)) >= min_translation
        else:
            # No odometry for this frame (or the previous one): fall back to
            # a minimum time spacing of 1 second.
            take = (t - last_t) >= _NO_ODOM_FALLBACK_DT_NS

        if take:
            selected[cam].append((path, sample))
            total += 1
            last_t = t
            last_pos = sample[0] if sample is not None else None

    if not selected["left"] and not selected["right"]:
        raise FileNotFoundError("No frames selected from the extract")

    if min_translation > 0 and num_frames > 0 and total < num_frames:
        print(
            f"  Note: displacement gate ({min_translation:.2f} m) exhausted the "
            f"extract: selected {total} of {len(timeline)} candidate frames. "
            "Use --min-translation 0 to process every frame."
        )

    return selected


def prepare_image_dir(frames: dict[str, list[Path]], work_dir: Path) -> Path:
    """Copy selected frames into a clean working image directory with camera prefixes."""
    img_dir = work_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    copy_jobs: list[tuple[Path, Path]] = []
    for cam_name, cam_frames in frames.items():
        for i, item in enumerate(cam_frames):
            img_path = item[0] if isinstance(item, tuple) else item
            copy_jobs.append((img_path, img_dir / f"{cam_name}_{i:06d}_{img_path.name}"))

    # Parallelize file copies (I/O bound); fall back to sequential on any failure.
    try:
        with ThreadPoolExecutor(max_workers=min(NUM_THREADS, max(1, len(copy_jobs)))) as pool:
            futures = [pool.submit(shutil.copy2, src, dst) for src, dst in copy_jobs]
            for fut in futures:
                fut.result()
    except Exception:
        for src, dst in copy_jobs:
            shutil.copy2(src, dst)

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
    try:
        return pycolmap.Database(str(db_path))
    except TypeError:
        return pycolmap.Database.open(str(db_path))


def _gpu_available() -> bool:
    """Detect whether the installed pycolmap build has a CUDA backend."""
    if not hasattr(pycolmap, "Device") or not hasattr(pycolmap, "has_cuda"):
        return False
    try:
        return bool(pycolmap.has_cuda)
    except Exception:
        return False


def run_sfm(
    image_dir: Path,
    output_dir: Path,
    refine_intrinsics: bool = False,
    pose_priors: dict[str, np.ndarray] | None = None,
    matching: str = "auto",
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

                if (
                    pose_priors is not None
                    and img_path.name in pose_priors
                    and hasattr(pycolmap, "PosePrior")
                    and hasattr(pycolmap, "PosePriorCoordinateSystem")
                ):
                    prior = pycolmap.PosePrior(
                        position=np.asarray(pose_priors[img_path.name], dtype=np.float64),
                        coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                    )
                    db.write_pose_prior(image_id, prior)

                if hasattr(db, "write_frame") and hasattr(pycolmap, "Frame"):
                    frame = pycolmap.Frame()
                    frame.rig_id = rig_ids[cam_name]
                    frame.add_data_id(image.data_id)
                    frame_id = db.write_frame(frame)
                    image.frame_id = frame_id
                    db.update_image(image)

    # ---- Feature extraction (GPU-accelerated when CUDA is available) ----
    print("Extracting features...")
    use_gpu = _gpu_available()
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.auto
    if use_gpu:
        print("  Using GPU for SIFT feature extraction.")
    else:
        print(f"  GPU not available; using CPU with {NUM_THREADS} threads.")
    extract_kwargs: dict = {"device": device}
    if hasattr(pycolmap, "FeatureExtractionOptions"):  # pycolmap >= 4.x
        ext = pycolmap.FeatureExtractionOptions()
        ext.num_threads = NUM_THREADS
        ext.use_gpu = use_gpu
        ext.sift.use_gpu = use_gpu
        extract_kwargs["extraction_options"] = ext
    else:  # pycolmap 3.x
        ext = pycolmap.SiftExtractionOptions()
        ext.num_threads = NUM_THREADS
        ext.use_gpu = use_gpu
        extract_kwargs["sift_options"] = ext
    pycolmap.extract_features(str(db_path), str(image_dir), **extract_kwargs)

    # ---- Feature matching ----
    print("Matching features...")
    if use_gpu:
        print("  Using GPU for feature matching.")
    num_images = sum(1 for _ in image_dir.iterdir()
                     if _.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if matching == "auto":
        matching = "sequential" if num_images > 150 else "exhaustive"
        print(f"  Auto mode: {num_images} images -> {matching} matching")
    match_kwargs: dict = {"device": device}
    if hasattr(pycolmap, "FeatureMatchingOptions"):  # pycolmap >= 4.x
        mat = pycolmap.FeatureMatchingOptions()
        mat.num_threads = NUM_THREADS
        mat.use_gpu = use_gpu
        mat.sift.use_gpu = use_gpu
        match_kwargs["matching_options"] = mat
    else:  # pycolmap 3.x: matcher options live in sift_options
        mat = pycolmap.SiftMatchingOptions()
        mat.num_threads = NUM_THREADS
        mat.use_gpu = use_gpu
        match_kwargs["sift_options"] = mat
    if matching == "sequential":
        # Sequential matching is O(n) instead of O(n^2): required for large sets.
        if hasattr(pycolmap, "SequentialMatchingOptions"):  # pycolmap 3.x
            seq = pycolmap.SequentialMatchingOptions()
            seq.num_threads = NUM_THREADS
            seq.overlap = 10
            match_kwargs["matching_options"] = seq
            pycolmap.match_sequential(str(db_path), **match_kwargs)
        elif hasattr(pycolmap, "SequentialPairingOptions"):  # pycolmap >= 4.x
            seq = pycolmap.SequentialPairingOptions()
            seq.overlap = 10
            match_kwargs["pairing_options"] = seq
            pycolmap.match_sequential(str(db_path), **match_kwargs)
        else:
            print("  Sequential matching unavailable; falling back to exhaustive.")
            pycolmap.match_exhaustive(str(db_path), **match_kwargs)
    else:
        pycolmap.match_exhaustive(str(db_path), **match_kwargs)

    # ---- Incremental reconstruction ----
    print("Running incremental reconstruction...")
    manager = pycolmap.ReconstructionManager()
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = refine_intrinsics
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = refine_intrinsics
    if hasattr(options, "num_threads"):
        options.num_threads = NUM_THREADS
    if hasattr(options, "ba_global_num_threads"):
        options.ba_global_num_threads = NUM_THREADS
    if hasattr(options, "ba_local_num_threads"):
        options.ba_local_num_threads = NUM_THREADS
    if hasattr(options, "image_path"):
        options.image_path = str(image_dir)
    if pose_priors and hasattr(options, "use_prior_position"):
        options.use_prior_position = True
        if hasattr(options, "use_robust_loss_on_prior_position"):
            options.use_robust_loss_on_prior_position = True

    if hasattr(pycolmap, "DatabaseCacheOptions"):  # pycolmap >= 4.x
        with _open_database(db_path) as db:
            cache = pycolmap.DatabaseCache.create(db, pycolmap.DatabaseCacheOptions())
        pipeline = pycolmap.IncrementalPipeline(options, cache, manager)
    else:  # pycolmap 3.x: pipeline takes paths directly
        pipeline = pycolmap.IncrementalPipeline(options, str(image_dir), str(db_path), manager)
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
        help=(
            "Total number of frames to select across both cameras; 0 = no cap "
            "(default: 50). Note: with displacement gating on, selection "
            "stops when the extract's odometry path is exhausted."
        ),
    )
    parser.add_argument(
        "--min-translation",
        type=float,
        default=1.0,
        help=(
            "Minimum odometry displacement in meters between consecutive selected "
            "frames (across both cameras) to keep bundle adjustments well "
            "conditioned. 0 disables the gate (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--use-odom-priors",
        action="store_true",
        help=(
            "Anchor camera positions to /odom wheel odometry pose priors "
            "(requires tf_static camera extrinsics in the extract)."
        ),
    )
    parser.add_argument(
        "--matching",
        choices=("auto", "exhaustive", "sequential"),
        default="auto",
        help=(
            "Feature matching strategy: exhaustive is O(n^2) and only viable "
            "for small sets; sequential is O(n) and required for full-folder "
            "runs (auto: exhaustive <= 150 images, sequential above)."
        ),
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

    odometry = load_odometry(args.extract)
    if args.min_translation > 0 and odometry is None:
        print(
            "warning: no /odom data found; displacement gating disabled.",
            file=sys.stderr,
        )

    n_target = "all" if args.num_frames == 0 else f"up to {args.num_frames}"
    print(f"Collecting {n_target} frames from {args.extract}")
    if args.min_translation > 0 and odometry is not None:
        print(f"  Requiring >= {args.min_translation:.2f} m odometry displacement between frames")
    frames = collect_frames(
        args.extract,
        args.num_frames,
        min_translation=args.min_translation,
        odometry=odometry,
    )
    print(f"  Left:  {len(frames['left'])} frames")
    print(f"  Right: {len(frames['right'])} frames")

    pose_priors: dict[str, np.ndarray] | None = None
    if args.use_odom_priors:
        extrinsics = load_tf_static_extrinsics(args.extract)
        pose_priors = {}
        for cam, cam_frames in frames.items():
            if cam not in extrinsics:
                continue
            _, t_base_cam = extrinsics[cam]
            for i, (path, sample) in enumerate(cam_frames):
                if sample is None:
                    continue
                base_pos, base_quat = sample
                R_odom_base = _quat_xyzw_to_rotmat(base_quat)
                cam_pos = base_pos + R_odom_base @ t_base_cam
                pose_priors[f"{cam}_{i:06d}_{path.name}"] = cam_pos
        if pose_priors:
            print(f"  Odometry pose priors: {len(pose_priors)} images")
        else:
            print("warning: --use-odom-priors set but no priors could be computed.", file=sys.stderr)
            pose_priors = None

    print(f"Preparing image directory in {args.output / 'images'}")
    img_dir = prepare_image_dir(frames, args.output)
    print(f"  {len(list(img_dir.iterdir()))} images copied")

    write_known_intrinsics(args.output)

    print("Running SfM pipeline...")
    manager = run_sfm(
        img_dir,
        args.output,
        refine_intrinsics=args.refine_intrinsics,
        pose_priors=pose_priors,
        matching=args.matching,
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
