#!/usr/bin/env python3
"""Run YOLO26 semantic segmentation on road images.

Uses the local yolo26m-sem checkpoint (Cityscapes). Class IDs come from
../public/dataset/data.yaml — you do not create that file from scratch, and you
do not need it to train before you can segment.

Writes `<images>/output/{semantic,road,overlays}/` next to the frames
(or next to a single image file).

Example:
    python yolo_segmenter.py ../public/dataset/images/f1_video1
    python yolo_segmenter.py ../public/gocart_extract/test4/images/left --max-images 10 -y
    python yolo_segmenter.py ../public/gocart_extract/test5/images/right --rewrite -y
    python yolo_segmenter.py frame.jpg --classes road -y
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "../public" / "models" / "yolo26m-sem"
DEFAULT_DATA = PROJECT_ROOT / "../public" / "dataset" / "data.yaml"

# Cityscapes palette in BGR for OpenCV overlays.
CITYSCAPES_PALETTE_BGR = {
    0: (128, 64, 128),
    1: (232, 35, 244),
    2: (70, 70, 70),
    3: (156, 102, 102),
    4: (153, 153, 190),
    5: (153, 153, 153),
    6: (30, 170, 250),
    7: (0, 220, 220),
    8: (35, 142, 107),
    9: (152, 251, 152),
    10: (180, 130, 70),
    11: (60, 20, 220),
    12: (0, 0, 255),
    13: (142, 0, 0),
    14: (70, 0, 0),
    15: (100, 60, 0),
    16: (100, 80, 0),
    17: (230, 0, 0),
    18: (32, 11, 119),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment road images with the local yolo26m-sem model."
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Image file, or a folder of images. Masks go in <images>/output/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"YOLO26-sem .pt file, or the unpacked model folder (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Dataset YAML used only for class names (default: {DEFAULT_DATA}).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["road"],
        help="Class names to export as binary masks (default: road).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference size (default: 1024). Raise this if road edges look blocky.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cpu or 0. Default: CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after this many images.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Blend weight for the color overlay (default: 0.45).",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Skip the model: fold sidewalk into road on existing output/semantic maps and rewrite road/overlays.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run without the confirmation prompt.",
    )
    return parser.parse_args()


def clip_dir(source: Path) -> Path:
    return source.parent if source.is_file() else source


def yolo_output_dir(source: Path) -> Path:
    return clip_dir(source) / "output"


def collect_images(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Images path not found: {source}")
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


def load_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}. "
            "It should live next to your dataset and list Cityscapes class names."
        )
    with data_yaml.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    raise ValueError(f"No `names` mapping in {data_yaml}")


def name_to_id(names: dict[int, str]) -> dict[str, int]:
    return {label.lower(): idx for idx, label in names.items()}


# Campus pavement is one surface. YOLO splits it across Cityscapes road and
# sidewalk in the same frame (test5 right_000075 / right_000164: interleaved
# blobs, sidewalk often the majority). Fold every alias into road.
ROAD_ALIASES = ("sidewalk",)


def fold_sidewalk_into_road(class_map: np.ndarray, lookup: dict[str, int]) -> np.ndarray:
    """Treat sidewalk as road, including mixed/chopped pavement blobs.

    Cityscapes often paints the same tarmac as road in one region and sidewalk
    in another. Remap every alias pixel to road so semantic, binary, and
    overlay outputs share one drivable surface.
    """
    road_id = lookup.get("road")
    if road_id is None:
        return class_map
    folded = class_map.copy()
    for name in ROAD_ALIASES:
        alias_id = lookup.get(name)
        if alias_id is None or alias_id == road_id:
            continue
        folded[folded == alias_id] = road_id
    return folded


def pack_unpacked_pt(model_dir: Path) -> Path:
    """The downloaded yolo26m-sem folder is an unzipped torch .pt checkpoint."""
    dest = model_dir.with_suffix(".pt")
    if dest.exists() and dest.stat().st_mtime >= model_dir.stat().st_mtime:
        return dest

    members = [path for path in model_dir.rglob("*") if path.is_file()]
    if not any(path.name == "data.pkl" for path in members):
        raise FileNotFoundError(
            f"{model_dir} is not a YOLO checkpoint (missing data.pkl)."
        )

    tmp = dest.with_suffix(".pt.tmp")
    prefix = dest.stem
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            rel = path.relative_to(model_dir).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{rel}")
            # Extracted checkpoints often keep the 1970-01-01 epoch, which zip rejects.
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, path.read_bytes())
    tmp.replace(dest)
    return dest


def resolve_model_path(model: Path) -> Path:
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if model.is_file():
        return model
    return pack_unpacked_pt(model)


def resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(model_path: Path, device: str):
    from ultralytics import YOLO

    return YOLO(str(model_path), task="semantic"), device


def class_map_from_result(result) -> np.ndarray:
    if getattr(result, "semantic_mask", None) is not None:
        data = result.semantic_mask.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        return np.asarray(data).squeeze().astype(np.uint8)

    # Older / instance-seg fallback: merge binary masks by class id.
    if result.masks is None or result.boxes is None:
        raise RuntimeError("Model returned neither semantic_mask nor instance masks.")
    h, w = result.orig_shape
    merged = np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data
    classes = result.boxes.cls
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if hasattr(classes, "cpu"):
        classes = classes.cpu().numpy()
    for mask, cls_id in zip(masks, classes):
        resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        merged[resized > 0] = int(cls_id)
    return merged


def colorize(class_map: np.ndarray) -> np.ndarray:
    color = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, bgr in CITYSCAPES_PALETTE_BGR.items():
        color[class_map == cls_id] = bgr
    return color


def confirm(n_images: int, rewrite: bool = False) -> bool:
    verb = "Rewrite" if rewrite else "Segment"
    try:
        answer = input(f"{verb} {n_images} image(s)? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def print_plan(
    images: list[Path],
    output: Path,
    model_path: Path | None,
    data_yaml: Path,
    names: dict[int, str],
    keep: list[str],
    imgsz: int,
    device: str,
    rewrite: bool = False,
) -> None:
    print(f"images:                    {images[0].parent if len(images) != 1 else images[0]}")
    print(f"image count:               {len(images)}")
    print(f"output folder:             {output}")
    print(f"mode:                      {'rewrite existing semantic maps' if rewrite else 'yolo inference'}")
    if model_path is not None:
        print(f"model:                     {model_path}")
    print(f"data.yaml:                 {data_yaml}")
    print(f"classes:                   {', '.join(f'{i}:{n}' for i, n in names.items())}")
    print(f"binary masks:              {', '.join(keep)}")
    print(f"ground aliases -> road:    {', '.join(ROAD_ALIASES)}")
    if not rewrite:
        print(f"imgsz / device:            {imgsz} / {device}")


def write_label_outputs(
    image_path: Path,
    class_map: np.ndarray,
    output: Path,
    *,
    keep_ids: dict[str, int],
    overlay_alpha: float,
    image: np.ndarray,
) -> None:
    stem = image_path.stem
    semantic_dir = output / "semantic"
    overlay_dir = output / "overlays"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(semantic_dir / f"{stem}.png"), class_map):
        raise RuntimeError(f"Failed to write semantic mask for {image_path.name}")

    overlay = cv2.addWeighted(image, 1.0 - overlay_alpha, colorize(class_map), overlay_alpha, 0)
    if not cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay):
        raise RuntimeError(f"Failed to write overlay for {image_path.name}")

    for name, cls_id in keep_ids.items():
        class_dir = output / name
        class_dir.mkdir(parents=True, exist_ok=True)
        binary = np.where(class_map == cls_id, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(class_dir / f"{stem}.png"), binary):
            raise RuntimeError(f"Failed to write {name} mask for {image_path.name}")


def load_and_fold_semantic(
    image_path: Path,
    semantic_path: Path,
    lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    class_map = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
    if class_map is None:
        raise RuntimeError(f"Could not read semantic mask: {semantic_path}")
    if class_map.ndim == 3:
        class_map = class_map[:, :, 0]
    if class_map.shape[:2] != image.shape[:2]:
        class_map = cv2.resize(
            class_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return image, fold_sidewalk_into_road(class_map, lookup)


def rewrite_jobs(images: list[Path], output: Path) -> list[tuple[Path, Path]]:
    jobs = []
    missing = 0
    for image_path in images:
        semantic_path = output / "semantic" / f"{image_path.stem}.png"
        if semantic_path.is_file():
            jobs.append((image_path, semantic_path))
        else:
            missing += 1
    if not jobs:
        raise FileNotFoundError(
            f"No semantic masks in {output / 'semantic'}. Run without --rewrite first."
        )
    if missing:
        print(f"note: skipping {missing} image(s) with no semantic mask")
    return jobs


def segment_image(
    model,
    image_path: Path,
    output: Path,
    *,
    keep_ids: dict[str, int],
    lookup: dict[str, int],
    imgsz: int,
    device: str,
    overlay_alpha: float,
) -> None:
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    class_map = class_map_from_result(results[0])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if class_map.shape[:2] != image.shape[:2]:
        class_map = cv2.resize(
            class_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    class_map = fold_sidewalk_into_road(class_map, lookup)
    write_label_outputs(
        image_path,
        class_map,
        output,
        keep_ids=keep_ids,
        overlay_alpha=overlay_alpha,
        image=image,
    )


def main() -> int:
    args = parse_args()

    try:
        images = collect_images(args.images)
        names = load_class_names(args.data)
        lookup = name_to_id(names)
        keep_ids = {}
        for label in args.classes:
            key = label.lower()
            if key not in lookup:
                known = ", ".join(names.values())
                raise ValueError(f"Unknown class {label!r}. Known names: {known}")
            keep_ids[key] = lookup[key]
        if "sidewalk" in keep_ids and "road" in lookup:
            keep_ids.pop("sidewalk")
            keep_ids.setdefault("road", lookup["road"])
        model_path = None if args.rewrite else resolve_model_path(args.model)
        device = "n/a" if args.rewrite else resolve_device(args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.max_images is not None:
        images = images[: args.max_images]

    output = yolo_output_dir(args.images)
    if args.rewrite:
        try:
            jobs = rewrite_jobs(images, output)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        images = [image_path for image_path, _ in jobs]
    else:
        jobs = []
        output.mkdir(parents=True, exist_ok=True)

    print_plan(
        images,
        output,
        model_path,
        args.data,
        names,
        list(keep_ids),
        args.imgsz,
        device,
        rewrite=args.rewrite,
    )

    if not images:
        print("nothing to segment.")
        return 0
    if not args.yes and not confirm(len(images), rewrite=args.rewrite):
        print("skipped. re-run with -y to continue.")
        return 0

    if args.rewrite:
        for i, (image_path, semantic_path) in enumerate(jobs, start=1):
            try:
                image, class_map = load_and_fold_semantic(image_path, semantic_path, lookup)
                write_label_outputs(
                    image_path,
                    class_map,
                    output,
                    keep_ids=keep_ids,
                    overlay_alpha=args.overlay_alpha,
                    image=image,
                )
            except Exception as exc:
                print(f"error: {image_path.name}: {exc}", file=sys.stderr)
                return 1
            if i == 1 or i == len(jobs) or i % 50 == 0:
                print(f"{i}/{len(jobs)}  {image_path.name}")
        print(f"done. rewrote {len(jobs)} image(s) -> {output}")
        print(f"  semantic class maps: {output / 'semantic'}")
        for name in keep_ids:
            print(f"  {name} binary masks:  {output / name}")
        print(f"  overlays:            {output / 'overlays'}")
        return 0

    try:
        model, device = load_model(model_path, device)
    except Exception as exc:  # ultralytics raises several types on a bad ckpt
        print(f"error: could not load model {model_path}: {exc}", file=sys.stderr)
        return 1

    for i, image_path in enumerate(images, start=1):
        try:
            segment_image(
                model,
                image_path,
                output,
                keep_ids=keep_ids,
                lookup=lookup,
                imgsz=args.imgsz,
                device=device,
                overlay_alpha=args.overlay_alpha,
            )
        except Exception as exc:
            print(f"error: {image_path.name}: {exc}", file=sys.stderr)
            return 1
        if i == 1 or i == len(images) or i % 25 == 0:
            print(f"{i}/{len(images)}  {image_path.name}")

    print(f"done. {len(images)} image(s) -> {output}")
    print(f"  semantic class maps: {output / 'semantic'}")
    for name in keep_ids:
        print(f"  {name} binary masks:  {output / name}")
    print(f"  overlays:            {output / 'overlays'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
