#!/usr/bin/env python3
"""ZED extract → combined left+right drivable BEV mosaic.

    python zed_mapper/run.py public/gocart_extract/test4 zed_mapper/out/test4 --max-frames 80 --debug -y

Auto-discovers left.mp4 + right.mp4 (and images/left + images/right) and fuses
both cameras into a single odom-frame map. Depth is only registered to the
left camera; the right camera contributes via its YOLO road mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZED X left+right drivable BEV mosaic from a gocart extract.")
    p.add_argument(
        "extract",
        type=Path,
        help="Folder with left.mp4, right.mp4, depth.mp4, pose_tum.txt, imu.csv, calib.yaml, plus images/left[/right].",
    )
    p.add_argument("output", type=Path, help="Folder for overlays, bev, map.png, drive.mp4 (combined output).")
    p.add_argument("--start", type=int, default=0, help="Skip this many frames at the start of the extract.")
    p.add_argument(
        "--masks",
        type=Path,
        default=None,
        help="Folder of left_XXXXXX.png road masks (default: <extract>/images/left/output/road). Right masks are auto-discovered at images/right/output/road.",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--ahead",
        type=float,
        default=8.0,
        help="Metres ahead of the cart to stamp into the map (keeps the path tight).",
    )
    p.add_argument("--fps", type=float, default=10.0, help="Output video frame rate (extracts are ~4 fps).")
    p.add_argument("--no-video", action="store_true", help="Skip writing drive.mp4.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument(
        "--mode",
        choices=["planar", "yolo", "hybrid"],
        default="hybrid",
        help="Drivable segmentation mode: planar, yolo, or hybrid (planar + validated YOLO union, default).",
    )
    return p.parse_args()


def confirm(n: int) -> bool:
    try:
        return input(f"Run zed_mapper on {n} frame(s)? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bev import Grid
    from dataset import ExtractSequence
    from mosaic import Mosaic
    from pipeline import process_frame
    from vis import compose_hud

    try:
        seq = ExtractSequence(args.extract, mask_dir=args.masks)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start = max(0, args.start)
    n = seq.n_frames - start if args.max_frames is None else min(seq.n_frames - start, args.max_frames)
    n = max(0, n)
    print(f"extract:      {args.extract}")
    print(f"frames:       {n}/{seq.n_frames}  start={start}")
    print(f"camera L:     {seq.camera_left.width}x{seq.camera_left.height}  fx={seq.camera_left.fx:.1f}")
    if seq.camera_right is not None:
        print(f"camera R:     {seq.camera_right.width}x{seq.camera_right.height}  fx={seq.camera_right.fx:.1f}  (no depth; YOLO-only)")
    else:
        print(f"camera R:     none (left-only extract)")
    imu_note = f"{seq.n_imu} IMU gyro samples" if seq.n_imu else "no imu.csv (VIO slerp only)"
    print(f"pose:         {seq.n_tum} TUM VIO + {imu_note}")
    print(f"road masks L: {seq.yolo_dir_left if seq.yolo_dir_left else 'none (asphalt prior)'}")
    print(f"road masks R: {seq.yolo_dir_right if seq.yolo_dir_right else 'none'}")
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

    grid = Grid(xmax=max(args.ahead + 4.0, 10.0))
    mosaic = Mosaic(stamp_xmax=args.ahead)
    prev_fit = None
    prev_plane = None
    processed = fitted = 0
    writer = None
    video_path = args.output / "drive.mp4"

    try:
        for frame in seq.iter_frames(args.max_frames, start=start):
            result = process_frame(
                frame, seq.camera_left, seq.camera_right, grid,
                prev_fit, prev_plane, mode=args.mode,
            )
            mosaic.add(frame.T_odom_cam, result.drive_bev, result.obst_bev, grid, result.fit)
            hud = compose_hud(result.overlay, mosaic.render(), result.bev_vis)
            stem = f"left_{frame.index:06d}"
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
                frame_out = hud
                if frame_out.shape[1] % 2 or frame_out.shape[0] % 2:
                    frame_out = frame_out[:vh, :vw]
                writer.write(frame_out)
            processed += 1
            if result.fit is not None:
                fitted += 1
                prev_fit = result.fit
            if result.plane is not None:
                prev_plane = result.plane
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
    if not args.no_video and processed:
        print(f"  video:    {video_path}  ({processed} frames @ {args.fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
