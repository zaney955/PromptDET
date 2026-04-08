from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .letterbox import letterbox_image
from .yolo_io import imread_rgb

CACHE_VERSION = 1


def _normalize_cache_root(cache_dir: str | Path, crop_size: int) -> Path:
    return Path(cache_dir).expanduser().resolve() / "prompt_crops" / f"cropsz_{crop_size}"


def _cache_key(image_path: str | Path, box: torch.Tensor, crop_size: int) -> str:
    resolved = str(Path(image_path).resolve())
    coords = [f"{float(value):.4f}" for value in box.tolist()]
    payload = "|".join([resolved, *coords, str(int(crop_size))])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def get_prompt_crop_cache_paths(
    cache_dir: str | Path,
    image_path: str | Path,
    box: torch.Tensor,
    crop_size: int,
) -> tuple[Path, Path]:
    root = _normalize_cache_root(cache_dir, crop_size)
    key = _cache_key(image_path, box, crop_size)
    return root / f"{key}.npy", root / f"{key}.json"


def _build_metadata(image_path: Path, box: torch.Tensor, crop_size: int) -> dict[str, int | str | list[float]]:
    stat = image_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "source_path": str(image_path.resolve()),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "crop_size": int(crop_size),
        "box": [round(float(value), 4) for value in box.tolist()],
    }


def is_prompt_crop_cache_stale(
    cache_dir: str | Path,
    image_path: str | Path,
    box: torch.Tensor,
    crop_size: int,
) -> bool:
    image_path = Path(image_path).resolve()
    cache_path, meta_path = get_prompt_crop_cache_paths(cache_dir, image_path, box, crop_size)
    if not cache_path.exists() or not meta_path.exists():
        return True
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    expected = _build_metadata(image_path, box, crop_size)
    for key, value in expected.items():
        if metadata.get(key) != value:
            return True
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception:
        return True
    return cached.shape != (crop_size, crop_size, 3) or cached.dtype != np.uint8


def load_or_create_prompt_crop(
    cache_dir: str | Path,
    image_path: str | Path,
    box: torch.Tensor,
    crop_size: int,
) -> np.ndarray:
    image_path = Path(image_path).resolve()
    cache_path, meta_path = get_prompt_crop_cache_paths(cache_dir, image_path, box, crop_size)
    if cache_path.exists() and meta_path.exists() and not is_prompt_crop_cache_stale(cache_dir, image_path, box, crop_size):
        return np.load(cache_path, allow_pickle=False)

    image = imread_rgb(image_path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.tolist()
    x1 = int(max(0, min(round(x1), width - 1)))
    y1 = int(max(0, min(round(y1), height - 1)))
    x2 = int(max(x1 + 1, min(round(x2), width)))
    y2 = int(max(y1 + 1, min(round(y2), height)))
    crop = np.ascontiguousarray(image[y1:y2, x1:x2], dtype=np.uint8)
    crop, _ = letterbox_image(crop, crop_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.ascontiguousarray(crop, dtype=np.uint8), allow_pickle=False)
    meta_path.write_text(
        json.dumps(_build_metadata(image_path, box, crop_size), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return crop
