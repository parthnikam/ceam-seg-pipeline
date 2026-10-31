#!/usr/bin/env python3
"""Lift YOLO road masks into a bird's-eye grid (Lift-Splat geometry).

Expects the yolo_segmenter layout: frames in `<source>/`, masks in
`<source>/output/{semantic,road}/`. Writes BEV next to that as
`<source>/lift_splat/`.

Example:
    python lift_splat/run.py public/dataset/images/f1_video1 --max-images 8 -y
    python lift_splat/run.py public/gocart_extract/test4/images/left --max-images 8 -y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift YOLO road masks into a bird's-eye occupancy grid."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Image folder (frames + output/ from yolo_segmenter), or a single image.",
    )
    parser.add_argument(
        "--occupancy",
        choices=("mask", "semantic"),
        default="semantic",
        help="Occupancy source. Default 'semantic' unions --class-ids (YOLO often labels F1 tarmac as wall). Use 'mask' for the binary road PNG.",
    )
    parser.add_argument(
        "--class-ids",
        default="0,3,9",
        help="Semantic class ids to treat as ground when --occupancy semantic (default: road,wall,terrain).",
    )
    parser.add_argument("--hfov", type=float, default=80.0, help="Horizontal FOV in degrees.")
    parser.add_argument("--height", type=float, default=1.15, help="Camera height in metres.")
    parser.add_argument("--pitch", type=float, default=12.0, help="Downward pitch in degrees.")
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.32,
        help="Ignore rows above this height fraction (sky / horizon smear).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.62,
        help="Ignore rows below this height fraction (cockpit).",
    )
    parser.add_argument(
        "--depth",
        choices=("ground", "uniform"),
        default="ground",
        help="Depth distribution along each ray. ground = peaked at z=0 (default).",
    )
    parser.add_argument("--final-h", type=int, default=288, help="Working image height (divisible by downsample).")
    parser.add_argument("--final-w", type=int, default=512, help="Working image width (divisible by downsample).")
    parser.add_argument("--downsample", type=int, default=4, help="Frustum stride in the working image.")
    parser.add_argument("--max-images", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, or cuda:0).")
    parser.add_argument("-y", "--yes", action="store_true", help="Run without the confirmation prompt.")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {source.suffix}")
        return [source]
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return images


def clip_dir(source: Path) -> Path:
    return source.parent if source.is_file() else source


def yolo_output_dir(source: Path) -> Path:
    return clip_dir(source) / "output"


def lift_output_dir(source: Path) -> Path:
    return clip_dir(source) / "lift_splat"


def resolve_label(image: Path, folder: Path, kind: str) -> Path:
    direct = folder / f"{image.stem}.png"
    if direct.exists():
        return direct
    nested = folder / kind / f"{image.stem}.png"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"No {kind} mask for {image.name} in {folder}. "
        "Run yolo_segmenter.py on this clip first."
    )


def confirm(n: int) -> bool:
    try:
        answer = input(f"Lift-splat {n} frame(s) into BEV? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def load_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask {path}")
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def load_semantic(path: Path, width: int, height: int, class_ids: set[int]) -> np.ndarray:
    sem = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if sem is None:
        raise RuntimeError(f"Could not read semantic mask {path}")
    if sem.shape[1] != width or sem.shape[0] != height:
        sem = cv2.resize(sem, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.isin(sem, list(class_ids)).astype(np.float32)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT))
    from lift_splat.camera import Camera
    from lift_splat.lift import GridConf, LiftSplat
    from lift_splat.visualize import colour_to_bgr, occupancy_to_bgr, panel

    try:
        images = collect_images(args.source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.max_images is not None:
        images = images[: args.max_images]
    yolo_dir = yolo_output_dir(args.source)
    mask_dir = yolo_dir / "road"
    semantic_dir = yolo_dir / "semantic"
    output = lift_output_dir(args.source)
    try:
        class_ids = {int(part) for part in args.class_ids.split(",") if part.strip() != ""}
    except ValueError:
        print(f"error: --class-ids must be comma-separated integers, got {args.class_ids!r}", file=sys.stderr)
        return 1
    if args.occupancy == "semantic" and not class_ids:
        print("error: --occupancy semantic needs at least one --class-ids value", file=sys.stderr)
        return 1

    if not images:
        print("nothing to process.")
        return 0
    try:
        if args.occupancy == "semantic":
            resolve_label(images[0], semantic_dir, "semantic")
        else:
            resolve_label(images[0], mask_dir, "road")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"source:        {args.source}")
    print(f"frame count:   {len(images)}")
    print(f"occupancy:     {args.occupancy}")
    print(f"masks:         {mask_dir if args.occupancy == 'mask' else semantic_dir}")
    print(f"output folder: {output}")
    print(
        f"camera:        hfov={args.hfov:.1f}deg  height={args.height:.2f}m  "
        f"pitch={args.pitch:.1f}deg down"
    )
    print(f"depth:         {args.depth}   y={args.y_min:.2f}..{args.y_max:.2f}")

    if not args.yes and not confirm(len(images)):
        print("skipped. re-run with -y to process.")
        return 0

    if args.final_h % args.downsample or args.final_w % args.downsample:
        print(
            f"error: --final-h and --final-w must be divisible by --downsample "
            f"({args.final_h}, {args.final_w}, {args.downsample})",
            file=sys.stderr,
        )
        return 1

    camera = Camera(
        width=args.final_w,
        height=args.final_h,
        hfov_deg=args.hfov,
        height_m=args.height,
        pitch_deg=args.pitch,
    )
    hit = camera.optical_axis_ground_m()
    if hit is not None:
        print(f"optical axis hits ground {hit:.1f}m ahead of the camera")

    grid = GridConf()
    model = LiftSplat(
        camera,
        grid=grid,
        downsample=args.downsample,
        device=args.device,
    )

    occ_dir = output / "bev"
    color_dir = output / "bev_color"
    overlay_dir = output / "overlays"
    for folder in (occ_dir, color_dir, overlay_dir):
        folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    try:
        for path in images:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"error: could not read {path}", file=sys.stderr)
                return 1
            working = cv2.resize(frame, (args.final_w, args.final_h), interpolation=cv2.INTER_AREA)
            if args.occupancy == "semantic":
                label_path = resolve_label(path, semantic_dir, "semantic")
                mask = load_semantic(label_path, args.final_w, args.final_h, class_ids)
            else:
                label_path = resolve_label(path, mask_dir, "road")
                mask = load_mask(label_path, args.final_w, args.final_h)
            rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            occupancy, colour = model.lift(
                mask,
                rgb=rgb,
                depth_mode=args.depth,
                y_max=args.y_max,
                y_min=args.y_min,
            )
            bev = occupancy_to_bgr(occupancy, grid)
            bev_color = colour_to_bgr(colour, grid)
            overlay = panel(frame, bev_color, mask)
            stem = path.stem
            if not cv2.imwrite(str(occ_dir / f"{stem}.png"), bev):
                raise RuntimeError(f"Failed to write BEV for {stem}")
            cv2.imwrite(str(color_dir / f"{stem}.png"), bev_color)
            cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay)
            processed += 1
            if processed == 1 or processed == len(images) or processed % 20 == 0:
                peak = float(occupancy.max()) if occupancy.size else 0.0
                print(f"{processed}/{len(images)}  {stem}  peak={peak:.2f}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"done. {processed} frame(s) -> {output}")
    print(f"  bev:      {occ_dir}")
    print(f"  colour:   {color_dir}")
    print(f"  overlays: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
