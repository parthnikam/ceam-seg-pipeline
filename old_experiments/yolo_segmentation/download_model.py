#!/usr/bin/env python3
"""Download the DeepLab Cityscapes checkpoint once and keep it on disk.

Example:
    python download_model.py
    python download_model.py --force
    python download_model.py --model xception65_cityscapes_trainfine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "../public" / "models" / "deeplab"

DEFAULT_MODEL_NAME = "mobilenetv2_coco_cityscapes_trainfine"
DOWNLOAD_URL_PREFIX = "http://download.tensorflow.org/models/"
MODEL_URLS = {
    "mobilenetv2_coco_cityscapes_trainfine": "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz",
    "xception65_cityscapes_trainfine": "deeplabv3_cityscapes_train_2018_02_06.tar.gz",
}


def tarball_path(model_name: str = DEFAULT_MODEL_NAME, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    if model_name not in MODEL_URLS:
        known = ", ".join(MODEL_URLS)
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {known}")
    return model_dir / MODEL_URLS[model_name]


def download_model(
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: Path = DEFAULT_MODEL_DIR,
    force: bool = False,
) -> Path:
    dest = tarball_path(model_name, model_dir)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"model already present, skipping download: {dest}")
        return dest

    url = DOWNLOAD_URL_PREFIX + MODEL_URLS[model_name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"downloading {model_name}")
    print(f"  {url}")
    print(f"  -> {dest}")
    try:
        urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print("download completed.")
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the DeepLab model tarball once.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=sorted(MODEL_URLS),
        help=f"Checkpoint to fetch (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the tarball is already on disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = download_model(model_name=args.model, force=args.force)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
