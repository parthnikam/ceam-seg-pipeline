#!/usr/bin/env python3
"""allnew extract → odom+IMU mosaic built from YOLO road masks (no depth).

Pipeline for public/gocart_extract/allnew: left+right ZED-lean PNGs, poses
from /odom with IMU gyro heading, road masks from YOLO26-sem stored under
<allnew>/images/segmented/{left,right}. The ground plane comes from the
tf_static camera mounts instead of a depth RANSAC fit.

    python zed_mapper_new/run.py public/gocart_extract/allnew zed_mapper_new/out/allnew --max-frames 80 --debug -y

Segment first (if images/segmented is empty), then map:

    python zed_mapper_new/run.py public/gocart_extract/allnew zed_mapper_new/out/allnew --segment -y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.append(str(PROJECT_ROOT / "zed_mapper"))

from bev import Grid  # noqa: E402
from mosaic import Mosaic  # noqa: E402
from vis import compose_hud  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Odom+IMU BEV mosaic from allnew YOLO road masks (left + right cameras)."
    )
    p.add_argument("extract", type=Path, help="allnew extract folder.")
    p.add_argument("output", type=Path, help="Folder for overlays, bev, map.png, drive.mp4.")
    p.add_argument("--start", type=int, default=0, help="Skip this many left images.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--segmented",
        type=Path,
        default=None,
        help="Segmented-mask root with left/ and right/ subfolders. "
        "Default: <extract>/images/segmented.",
    )
    p.add_argument(
        "--ahead",
        type=float,
        default=8.0,
        help="Metres ahead of the cart to stamp into the map (keeps the path tight).",
    )
    p.add_argument(
        "--imu",
        choices=["auto", "livox", "imu", "none"],
        default="auto",
        help="IMU gyro source for heading between /odom samples. 'auto' tries /imu/data then "
        "/livox/imu and keeps whichever validates against /odom yaw.",
    )
    p.add_argument("--pair-tol", type=float, default=0.25, help="Max |dt| to pair a right image.")
    p.add_argument(
        "--cam-height",
        type=float,
        default=None,
        help="Override the tf_static camera height above ground (m), e.g. 0.56.",
    )
    p.add_argument(
        "--grid-width",
        type=float,
        default=9.0,
        help="Lateral half-width (m) of the ego BEV grid. Wide campus roads "
        "(road + sidewalk often > 12 m) need more than the default 6 m or their "
        "edge rows get dropped as FOV-clipped and the spline fit fails.",
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument(
        "--segment",
        action="store_true",
        help="Run the YOLO segmentation stage first (images/segmented/{left,right}).",
    )
    p.add_argument("--model", type=Path, default=None, help="YOLO model (with --segment).")
    p.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size (with --segment).")
    p.add_argument("--device", default=None, help="YOLO device (with --segment).")
    return p.parse_args()


def confirm(n: int) -> bool:
    try:
        return input(f"Run zed_mapper_new on {n} frame(s)? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def map_window(
    odom_xy: np.ndarray, ahead: float, grid_half_width: float = 6.0
) -> tuple[float, float, float, float, float]:
    """Map window (center_x, center_y, extent, width, height) from the driven path.

    /odom gives every position the cart took (the IMU only refines heading),
    so its x/y bounds plus a stamp margin cover the full driven corridor:
    every stamped cell lies within ahead + grid_half_width metres of a driven
    position (plus a small pad).
    """
    margin = ahead + grid_half_width + 2.0
    lo = odom_xy.min(axis=0) - margin
    hi = odom_xy.max(axis=0) + margin
    width, height = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    center_x, center_y = float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)
    extent = max(width, height)
    return center_x, center_y, extent, width, height


def main() -> int:
    args = parse_args()
    from dataset_new import AllnewSequence
    from ground import Plane
    from pipeline_new import process_frame_allnew
    from segment_new import DEFAULT_MODEL, segment_all

    if args.segment:
        try:
            segment_all(
                args.extract,
                model=args.model if args.model is not None else DEFAULT_MODEL,
                imgsz=args.imgsz,
                device=args.device,
                yes=args.yes,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"error: segmentation failed: {exc}", file=sys.stderr)
            return 1

    try:
        seq = AllnewSequence(
            args.extract,
            segmented_dir=args.segmented,
            imu=args.imu,
            pair_tol=args.pair_tol,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if seq.mask_count_left == 0 and seq.mask_count_right == 0:
        print(
            f"error: no road masks under {seq.mask_dir_left.parent}\n"
            "       run zed_mapper_new/segment_new.py first (or pass --segment).",
            file=sys.stderr,
        )
        return 1

    plane = seq.plane
    if args.cam_height is not None:
        plane = Plane(np.array([0.0, 0.0, 1.0]), float(args.cam_height))

    start = max(0, args.start)
    n = seq.n_frames - start if args.max_frames is None else min(seq.n_frames - start, args.max_frames)
    n = max(0, n)
    print(f"extract:      {args.extract}")
    print(f"camera L:     {seq.camera_left.width}x{seq.camera_left.height}  fx={seq.camera_left.fx:.1f}")
    if seq.camera_right is not None:
        print(f"camera R:     {seq.camera_right.width}x{seq.camera_right.height}  fx={seq.camera_right.fx:.1f}")
    print(f"frames:       {n}/{seq.n_frames}  start={start}  (right images: {seq.right_count})")
    print(
        f"plane:        tf_static mount, cam height {seq.cam_height_m:.3f} m, "
        f"baseline offset ({seq.baseline_offset[0]:+.2f}, {seq.baseline_offset[1]:+.2f}) m"
    )
    if args.cam_height is not None:
        print(f"plane:        overridden cam height {args.cam_height:.3f} m")
    print(f"pose:         {seq.n_odom} /odom samples "
          f"[{seq._odom_t[0]:.1f} .. {seq._odom_t[-1]:.1f}]")
    if seq.n_frames_past_odom:
        print(f"pose:         {seq.n_frames_past_odom} frame(s) past the last /odom sample "
              "(position held, gyro continues heading)")
    if seq.n_imu:
        err = seq.imu_yaw_err_rad
        err_txt = f" yaw-err {err:.2f} rad" if np.isfinite(err) else ""
        imu_label = {"imu": "/imu/data", "livox": "/livox/imu"}.get(seq.imu_source, seq.imu_source)
        print(f"imu:          {imu_label}  {seq.n_imu} samples{err_txt}")
        print(f"imu:          {seq.imu_note}")
    else:
        print("imu:          none; /odom quaternions slerped")
    print(f"masks L/R:    {seq.mask_count_left} / {seq.mask_count_right}  at {seq.mask_dir_left.parent}")
    print(f"stamp ahead:  {args.ahead:.1f} m")
    print(f"output:       {args.output}")
    if n == 0:
        print("nothing to process.")
        return 0
    if not args.yes and not confirm(n):
        print("skipped. re-run with -y to process.")
        return 0

    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (args.output / "bev").mkdir(parents=True, exist_ok=True)

    grid = Grid(
        xmax=max(args.ahead + 4.0, 10.0),
        ymin=-args.grid_width,
        ymax=args.grid_width,
    )
    cx, cy, extent, win_w, win_h = map_window(seq.odom_xy, args.ahead, args.grid_width)
    mosaic = Mosaic(stamp_xmax=args.ahead, extent_m=extent, center_x=cx, center_y=cy)
    print(
        f"map window:   {win_w:.0f} x {win_h:.0f} m from /odom path "
        f"(extent {extent:.0f} m, centre {cx:+.0f}, {cy:+.0f})"
    )
    prev_fit = None
    processed = fitted = 0
    missing_l = missing_r = 0
    writer = None
    video_path = args.output / "drive.mp4"
    vw = vh = 0

    try:
        for frame in seq.iter_frames(args.max_frames, start=start):
            result = process_frame_allnew(
                frame,
                seq.camera_left,
                seq.camera_right,
                grid,
                plane,
                prev_fit=prev_fit,
                baseline=seq.baseline_offset,
            )
            mosaic.add(frame.T_odom_cam, result.drive_bev, result.obst_bev, grid, result.fit)
            hud = compose_hud(result.overlay, mosaic.render(), result.bev_vis)
            stem = frame.stem
            if not cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), hud):
                raise RuntimeError(f"failed to write overlay {stem}")
            if args.debug:
                cv2.imwrite(str(args.output / "bev" / f"{stem}.png"), result.bev_vis)
            if not args.no_video:
                if writer is None:
                    vh, vw = hud.shape[:2]
                    vw -= vw % 2
                    vh -= vh % 2
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (vw, vh))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer {video_path}")
                frame_out = hud[:vh, :vw] if (hud.shape[0] % 2 or hud.shape[1] % 2) else hud
                writer.write(frame_out)
            processed += 1
            if frame.road_left is None:
                missing_l += 1
            if seq.camera_right is not None and frame.bgr_right is not None and frame.road_right is None:
                missing_r += 1
            if result.fit is not None:
                fitted += 1
                prev_fit = result.fit
            if processed in {1, n} or processed % 20 == 0:
                tag = "ok" if result.fit is not None else "no-fit"
                print(f"{processed}/{n}  {stem}  {tag}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if writer is not None:
            writer.release()
        return 1
    if writer is not None:
        writer.release()

    rendered = mosaic.render()
    map_png = args.output / "map.png"
    if not cv2.imwrite(str(map_png), rendered):
        print(f"error: failed to write {map_png}", file=sys.stderr)
        return 1
    print(f"done. fitted {fitted}/{processed} -> {args.output}")
    print(f"  overlays: {overlay_dir}")
    print(f"  map:      {map_png}")
    if missing_l or missing_r:
        print(f"  frames without masks: left {missing_l}, right {missing_r}")
    if not args.no_video and processed:
        print(f"  video:    {video_path}  ({processed} frames @ {args.fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
