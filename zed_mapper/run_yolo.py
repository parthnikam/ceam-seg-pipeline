#!/usr/bin/env python3
"""ZED extract → combined left+right BEV mosaic using YOLO road masks as the segmenter.

Does not use segment.fuse_drivable (planar flood-fill). The 2D drivable mask
is the YOLO PNG as-is, lifted with the same IPM / mosaic stack. Left camera
also contributes depth for ground-plane estimation; right camera is YOLO-only.

    python zed_mapper/run_yolo.py public/gocart_extract/test4 zed_mapper/out/test4_yolo --max-frames 80 --debug -y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from camera import Camera

DEFAULT_MASKS = Path("images") / "left" / "output" / "road"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZED mapper driven by YOLO road masks (left + right).")
    p.add_argument("extract", type=Path, help="Extract folder (test4 / test5).")
    p.add_argument("output", type=Path, help="Folder for overlays, bev, map.png, drive.mp4.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--masks",
        type=Path,
        default=None,
        help="Folder of left_XXXXXX.png masks. Default: <extract>/images/left/output/road.",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--ahead",
        type=float,
        default=8.0,
        help="Metres ahead of the cart to stamp into the map (keeps the path tight).",
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    return p.parse_args()


def confirm(n: int) -> bool:
    try:
        return input(f"Run YOLO-mask zed_mapper on {n} frame(s)? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def fuse_yolo(yolo_road: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Drivable = YOLO road mask. Empty obstacle layer (YOLO is the segmenter)."""
    h, w = shape
    if yolo_road is None:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    mask = yolo_road
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    drivable = (mask > 0).astype(np.uint8) * 255
    return drivable, np.zeros((h, w), np.uint8)


def process_frame_yolo(frame, camera_left: Camera, camera_right: Camera | None, grid: Grid,
                       prev_fit, prev_plane):
    """Process one stereo frame pair with YOLO as the segmenter for both cameras."""
    from bev import build_bev
    from boundaries import fit_track
    from ground import ransac_plane, sample_points
    from pipeline import FrameResult
    from vis import overlay_camera, render_bev

    bgr_left = frame.bgr_left
    bgr_right = frame.bgr_right
    depth_left = frame.depth_m_left
    yolo_left = frame.yolo_road_left
    yolo_right = frame.yolo_road_right

    # ---- Ground plane from left depth (shared by both cameras) ----
    pts = sample_points(camera_left, depth_left)
    plane = ransac_plane(pts) or prev_plane
    if plane is None:
        empty = np.zeros(bgr_left.shape[:2], np.uint8)
        bev0 = np.zeros((grid.nx, grid.ny), np.uint8)
        return FrameResult(
            empty, empty, bev0, bev0, None, None,
            bgr_left.copy(), render_bev(bev0, bev0, grid, None),
        )

    # ---- Left camera: YOLO mask → BEV ----
    drivable_l, _obstacle_l = fuse_yolo(yolo_left, bgr_left.shape[:2])
    drive_bev_l, obst_bev_l = build_bev(
        drivable_l, _obstacle_l, camera_left, plane, depth_left, grid
    )

    # ---- Right camera: YOLO mask → BEV (no depth, shared plane) ----
    drive_bev_r = np.zeros((grid.nx, grid.ny), np.uint8)
    obst_bev_r = np.zeros((grid.nx, grid.ny), np.uint8)
    if camera_right is not None and bgr_right is not None and yolo_right is not None:
        drivable_r, _obstacle_r = fuse_yolo(yolo_right, bgr_right.shape[:2])
        drive_bev_r, obst_bev_r = build_bev(
            drivable_r, _obstacle_r, camera_right, plane, None, grid
        )

    # ---- Union both cameras' BEVs ----
    drive_bev = np.bitwise_or(drive_bev_l, drive_bev_r)
    obst_bev = np.bitwise_or(obst_bev_l, obst_bev_r)

    fit = fit_track(drive_bev, grid, prev=prev_fit)

    return FrameResult(
        drivable_l,
        _obstacle_l,
        drive_bev,
        obst_bev,
        plane,
        fit,
        overlay_camera(bgr_left, drivable_l, _obstacle_l, camera_left, plane, fit),
        render_bev(drive_bev, obst_bev, grid, fit),
    )


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bev import Grid
    from dataset import ExtractSequence
    from mosaic import Mosaic
    from vis import compose_hud

    mask_dir = args.masks if args.masks is not None else args.extract / DEFAULT_MASKS
    try:
        seq = ExtractSequence(args.extract, mask_dir=mask_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if seq.yolo_dir_left is None:
        print(f"error: no YOLO road masks at {mask_dir}", file=sys.stderr)
        return 1

    start = max(0, args.start)
    n = seq.n_frames - start if args.max_frames is None else min(seq.n_frames - start, args.max_frames)
    n = max(0, n)
    print(f"extract:      {args.extract}")
    print(f"segmenter:    YOLO masks (left + right, not planar flood-fill)")
    print(f"camera L:     {seq.camera_left.width}x{seq.camera_left.height}  fx={seq.camera_left.fx:.1f}")
    if seq.camera_right is not None:
        print(f"camera R:     {seq.camera_right.width}x{seq.camera_right.height}  fx={seq.camera_right.fx:.1f}  (YOLO-only)")
    print(f"road masks L: {seq.yolo_dir_left}")
    print(f"road masks R: {seq.yolo_dir_right if seq.yolo_dir_right else 'none'}")
    print(f"frames:       {n}/{seq.n_frames}  start={start}")
    imu_note = f"{seq.n_imu} IMU gyro samples" if seq.n_imu else "no imu.csv (VIO slerp only)"
    print(f"pose:         {seq.n_tum} TUM VIO + {imu_note}")
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
            result = process_frame_yolo(
                frame, seq.camera_left, seq.camera_right, grid,
                prev_fit, prev_plane,
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
                frame_out = hud[:vh, :vw] if (hud.shape[0] % 2 or hud.shape[1] % 2) else hud
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
