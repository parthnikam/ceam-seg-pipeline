#!/usr/bin/env python3
"""YOLO26-sem road segmentation for the allnew extract's left/right images.

Writes binary road masks (255 = road) to `<allnew>/images/segmented/{left,right}/`
named after each source image stem, e.g.

    images/segmented/left/20260905_071723_285318.png

so zed_mapper_new/run.py can pick them up. Reuses the same model loading,
semantic-mask extraction, and sidewalk-into-road folding as
old_experiments/yolo_segmentation/yolo_segmenter.py.

    python zed_mapper_new/segment_new.py public/gocart_extract/allnew -y
    python zed_mapper_new/segment_new.py public/gocart_extract/allnew --max-images 40 -y
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
sys.path.append(str(PROJECT_ROOT / "old_experiments" / "yolo_segmentation"))

from dataset_new import CITYSCAPES_FALLBACK_NAMES  # noqa: E402
from yolo_segmenter import (  # noqa: E402
    CITYSCAPES_PALETTE_BGR,
    class_map_from_result,
    collect_images,
    colorize,
    fold_sidewalk_into_road,
    load_class_names,
    load_model,
    name_to_id,
    resolve_device,
    resolve_model_path,
)

DEFAULT_MODEL = PROJECT_ROOT / "public" / "models" / "yolo26x-sem.pt"
DEFAULT_DATA = PROJECT_ROOT / "public" / "dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment allnew left/right images with YOLO26-sem; masks -> images/segmented."
    )
    p.add_argument("extract", type=Path, help="allnew extract folder.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default=None, help="cpu, 0, ... Default: CUDA if available.")
    p.add_argument("--max-images", type=int, default=None, help="Per camera side.")
    p.add_argument("--rewrite", action="store_true", help="Re-segment even when the mask exists.")
    p.add_argument(
        "--overlays",
        action="store_true",
        help="Also write colorized semantic overlays to images/segmented/overlays.",
    )
    p.add_argument("-y", "--yes", action="store_true")
    return p.parse_args()


def _class_lookup(data: Path) -> dict[str, int]:
    try:
        return name_to_id(load_class_names(data))
    except FileNotFoundError:
        print(f"note: {data} not found; using built-in Cityscapes ids (road=0, sidewalk=1)")
        return {name: idx for idx, name in CITYSCAPES_FALLBACK_NAMES.items()}


def segment_all(
    extract: Path,
    model: Path | None = None,
    data: Path | None = None,
    imgsz: int = 1024,
    device: str | None = None,
    max_images: int | None = None,
    rewrite: bool = False,
    overlays: bool = False,
    yes: bool = False,
) -> tuple[int, int]:
    """Segment both camera sides; returns (n_left_masks, n_right_masks) written."""
    root = Path(extract)
    lookup = _class_lookup(Path(data) if data is not None else DEFAULT_DATA)
    road_id = lookup.get("road")
    if road_id is None:
        raise RuntimeError("class name 'road' not found in the dataset names")

    jobs: list[tuple[Path, Path]] = []
    for side in ("left", "right"):
        src = root / "images" / side
        if not src.is_dir():
            print(f"note: no images/{side}; skipping that side")
            continue
        out_dir = root / "images" / "segmented" / side
        images = collect_images(src)
        if max_images is not None:
            images = images[:max_images]
        for image_path in images:
            mask_path = out_dir / f"{image_path.stem}.png"
            if rewrite or not mask_path.is_file():
                jobs.append((image_path, mask_path))
    if not jobs:
        print("segmentation: all masks already present (use --rewrite to redo).")
        return 0, 0

    model_path = resolve_model_path(Path(model) if model is not None else DEFAULT_MODEL)
    device = resolve_device(device)
    print(f"segmentation: {len(jobs)} image(s)")
    print(f"  model:  {model_path}")
    print(f"  device: {device}  imgsz={imgsz}")
    print(f"  out:    {root / 'images' / 'segmented'}")
    if not yes:
        try:
            if input(f"Segment {len(jobs)} image(s)? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("skipped segmentation.")
                return 0, 0
        except EOFError:
            return 0, 0

    for side in ("left", "right"):
        (root / "images" / "segmented" / side).mkdir(parents=True, exist_ok=True)
    if overlays:
        for side in ("left", "right"):
            (root / "images" / "segmented" / "overlays" / side).mkdir(parents=True, exist_ok=True)

    yolo, device = load_model(model_path, device)
    written = {"left": 0, "right": 0}
    for i, (image_path, mask_path) in enumerate(jobs, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {image_path}")
        results = yolo.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)
        class_map = class_map_from_result(results[0])
        if class_map.shape[:2] != image.shape[:2]:
            class_map = cv2.resize(class_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        class_map = fold_sidewalk_into_road(class_map, lookup)
        binary = np.where(class_map == road_id, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(mask_path), binary):
            raise RuntimeError(f"failed to write {mask_path}")
        written["left" if "left" in image_path.parts else "right"] += 1
        if overlays:
            overlay_dir = root / "images" / "segmented" / "overlays" / (
                "left" if "left" in image_path.parts else "right"
            )
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay = cv2.addWeighted(image, 0.55, colorize(class_map), 0.45, 0)
            cv2.imwrite(str(overlay_dir / f"{image_path.stem}.jpg"), overlay)
        if i == 1 or i == len(jobs) or i % 25 == 0:
            print(f"  {i}/{len(jobs)}  {image_path.name}")
    print(f"segmentation done: {written['left']} left + {written['right']} right masks written.")
    return written["left"], written["right"]


def main() -> int:
    args = parse_args()
    try:
        segment_all(
            args.extract,
            model=args.model,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            max_images=args.max_images,
            rewrite=args.rewrite,
            overlays=args.overlays,
            yes=args.yes,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
