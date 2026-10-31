#!/usr/bin/env python3
"""White paint lines -> driveable BEV -> overlapping top-down map.

    python run.py ../public/dataset/videos/f1_video1.mp4 out/f1_video1 --max-images 80 --debug -y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="White-line driveable BEV, stitched across frames.")
    p.add_argument("source", type=Path, help="Image, image folder, or video.")
    p.add_argument("output", type=Path, help="Folder for overlays and map.png.")
    p.add_argument("--hfov", type=float, default=80.0)
    p.add_argument("--height", type=float, default=1.15)
    p.add_argument("--pitch", type=float, default=16.0)
    p.add_argument("--y-min", type=float, default=0.28)
    p.add_argument("--y-max", type=float, default=0.48)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    return p.parse_args()


def collect_frames(source: Path) -> tuple[str, list[Path]]:
    """Return ('video'|'images', paths)."""
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    suf = source.suffix.lower()
    if source.is_file() and suf in VIDEO_SUFFIXES:
        return "video", [source]
    if source.is_file() and suf in IMAGE_SUFFIXES:
        return "images", [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise FileNotFoundError(f"No images found in {source}")
        return "images", images
    raise ValueError(f"Unsupported source: {source}")


def iter_video(path: Path, max_images: int | None):
    """Yield (stem, bgr) for each video frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    index = 0
    try:
        while max_images is None or index < max_images:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{path.stem}_{index:06d}", frame
            index += 1
    finally:
        cap.release()


def confirm(n: int) -> bool:
    try:
        return input(f"Run white-line BEV pipeline on {n} frame(s)? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camera import Camera
    from ground import Grid, Mosaic
    from pipeline import process_frame

    try:
        kind, items = collect_frames(args.source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if kind == "images":
        if args.max_images is not None:
            items = items[: args.max_images]
        count = len(items)
    else:
        cap = cv2.VideoCapture(str(items[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        count = total if args.max_images is None else min(total, args.max_images)

    print(f"source:        {args.source}")
    print(f"frame count:   {count}")
    print(f"output folder: {args.output}")
    print(f"camera:        hfov={args.hfov:.1f}deg  height={args.height:.2f}m  pitch={args.pitch:.1f}deg")
    print(f"roi band:      y={args.y_min:.2f} .. {args.y_max:.2f}")
    if count == 0:
        print("nothing to process.")
        return 0
    if not args.yes and not confirm(count):
        print("skipped. re-run with -y to process.")
        return 0

    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (args.output / "paint").mkdir(parents=True, exist_ok=True)
        (args.output / "bev").mkdir(parents=True, exist_ok=True)

    camera = Camera(1920, 1080, args.hfov, args.height, args.pitch)
    grid = Grid()
    mosaic = Mosaic()
    prev_fit = None
    processed = found = 0

    def handle(stem: str, frame) -> None:
        nonlocal camera, prev_fit, processed, found
        h, w = frame.shape[:2]
        if camera.width != w or camera.height != h:
            camera = Camera(w, h, args.hfov, args.height, args.pitch)
        result = process_frame(frame, camera, grid, prev_fit, args.y_min, args.y_max)
        mosaic.add(
            frame, result.driveable, result.driveable_bev, result.paint_bev,
            result.fit, camera, grid, args.y_min, args.y_max,
        )
        if not cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), result.overlay):
            raise RuntimeError(f"Failed to write overlay for {stem}")
        if args.debug:
            cv2.imwrite(str(args.output / "paint" / f"{stem}.png"), result.paint)
            cv2.imwrite(str(args.output / "bev" / f"{stem}.png"), result.bev_vis)
        processed += 1
        if result.fit is not None:
            found += 1
            prev_fit = result.fit
        if processed in {1, count} or processed % 25 == 0:
            print(f"{processed}/{count}  {stem}  {'ok' if result.fit else 'no-fit'}")

    try:
        if kind == "images":
            for path in items:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"error: could not read {path}", file=sys.stderr)
                    return 1
                handle(path.stem, frame)
        else:
            for stem, frame in iter_video(items[0], args.max_images):
                handle(stem, frame)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    map_path = args.output / "map.png"
    if not cv2.imwrite(str(map_path), mosaic.render()):
        print(f"error: failed to write {map_path}", file=sys.stderr)
        return 1
    print(f"done. fitted {found}/{processed} frame(s) -> {args.output}")
    print(f"  overlays: {overlay_dir}")
    print(f"  map:      {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
