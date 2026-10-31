#!/usr/bin/env python3
"""Extract image frames from a single video.

Shows total frames and how many will be written, then waits for opt-in.
Use -y to skip the confirmation prompt.

Example:
    python extract_frames.py public/dataset/videos/f1_video1.mp4 public/dataset/images
    python extract_frames.py clip.mp4 ./frames --fps 2 -y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a video into individual image frames."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the video file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Folder to write image frames into.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth frame (default: 1, i.e. every frame).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target extraction rate in frames per second. Overrides --every.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: end of video).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after writing this many frames.",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="Output image format (default: jpg).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95). Ignored for PNG.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix. Defaults to the video's stem name.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Extract without the confirmation prompt.",
    )
    return parser.parse_args()


def validate_video(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a video file, got: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video type: {path.suffix}")


def encode_params(fmt: str, quality: int) -> list[int]:
    if fmt == "jpg":
        quality = max(1, min(100, quality))
        return [cv2.IMWRITE_JPEG_QUALITY, quality]
    return [cv2.IMWRITE_PNG_COMPRESSION, 3]


def stride_from_fps(video_fps: float, target_fps: float) -> int:
    if video_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, round(video_fps / target_fps))


def planned_extract_count(
    total: int,
    stride: int,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
) -> int:
    span = max(0, min(end_frame, total) - max(0, start_frame))
    if span == 0 or stride < 1:
        return 0
    planned = (span + stride - 1) // stride
    if max_frames is not None:
        planned = min(planned, max_frames)
    return planned


def inspect_video(
    video_path: Path,
    *,
    every: int,
    target_fps: float | None,
    start_s: float,
    end_s: float | None,
    max_frames: int | None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    stride = stride_from_fps(video_fps, target_fps) if target_fps else max(1, every)
    start_frame = int(start_s * video_fps) if video_fps > 0 else 0
    if end_s is not None and video_fps > 0:
        end_frame = int(end_s * video_fps)
    else:
        end_frame = total
    duration = (total / video_fps) if video_fps > 0 else 0.0
    to_extract = planned_extract_count(
        total, stride, start_frame, end_frame, max_frames
    )
    return {
        "fps": video_fps,
        "total": total,
        "duration": duration,
        "stride": stride,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "to_extract": to_extract,
    }


def confirm_extract(planned_total: int) -> bool:
    prompt = f"Extract {planned_total} frame(s)? [y/N] "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    info: dict,
    max_frames: int | None,
    fmt: str,
    quality: int,
    prefix: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    start_frame = info["start_frame"]
    end_frame = info["end_frame"]
    stride = info["stride"]

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    params = encode_params(fmt, quality)
    saved = 0
    index = start_frame
    try:
        while True:
            if end_frame and index >= end_frame:
                break
            if max_frames is not None and saved >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if (index - start_frame) % stride == 0:
                dest = output_dir / f"{prefix}_{saved:06d}.{fmt}"
                if not cv2.imwrite(str(dest), frame, params):
                    raise RuntimeError(f"Failed to write {dest}")
                saved += 1

            index += 1
    finally:
        cap.release()

    return saved


def print_plan(video: Path, output_dir: Path, info: dict) -> None:
    print(f"video:                     {video}")
    print(f"output folder:             {output_dir}")
    print(f"total frames in video:     {info['total']}")
    print(f"frames to be extracted:    {info['to_extract']}")
    print(
        f"source:                    {info['fps']:.2f} fps, "
        f"{info['duration']:.1f}s, stride={info['stride']}"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_video(args.video)
        info = inspect_video(
            args.video,
            every=args.every,
            target_fps=args.fps,
            start_s=args.start,
            end_s=args.end,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print_plan(args.video, args.output, info)

    if info["to_extract"] == 0:
        print("nothing to extract.")
        return 0

    if not args.yes and not confirm_extract(info["to_extract"]):
        print("skipped. output folder is ready; re-run with -y to extract.")
        return 0

    prefix = args.prefix or args.video.stem
    try:
        saved = extract_frames(
            args.video,
            args.output,
            info=info,
            max_frames=args.max_frames,
            fmt=args.format,
            quality=args.quality,
            prefix=prefix,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"done. wrote {saved} of {info['to_extract']} planned "
        f"({info['total']} total in video) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
