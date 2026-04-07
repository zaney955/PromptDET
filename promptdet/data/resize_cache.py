from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .letterbox import letterbox_image
from .yolo_io import imread_rgb

CACHE_VERSION = 2


def _normalize_cache_root(cache_dir: str | Path, image_size: int) -> Path:
    return Path(cache_dir).expanduser().resolve() / f"imgsz_{image_size}"


def _cache_key(image_path: str | Path) -> str:
    return hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()


def get_resize_cache_paths(cache_dir: str | Path, image_path: str | Path, image_size: int) -> tuple[Path, Path]:
    root = _normalize_cache_root(cache_dir, image_size)
    key = _cache_key(image_path)
    return root / f"{key}.npy", root / f"{key}.json"


def _build_metadata(image_path: Path, image_size: int) -> dict[str, int | str]:
    stat = image_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "source_path": str(image_path.resolve()),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "image_size": int(image_size),
    }


def is_resize_cache_stale(cache_dir: str | Path, image_path: str | Path, image_size: int) -> bool:
    image_path = Path(image_path).resolve()
    cache_path, meta_path = get_resize_cache_paths(cache_dir, image_path, image_size)
    if not cache_path.exists() or not meta_path.exists():
        return True
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    expected = _build_metadata(image_path, image_size)
    for key, value in expected.items():
        if metadata.get(key) != value:
            return True
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception:
        return True
    return cached.shape != (image_size, image_size, 3) or cached.dtype != np.uint8


def _write_resize_cache(cache_dir: str | Path, image_path: str | Path, image_size: int) -> str:
    image_path = Path(image_path).resolve()
    cache_path, meta_path = get_resize_cache_paths(cache_dir, image_path, image_size)
    existed = cache_path.exists() and meta_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resized, _ = letterbox_image(imread_rgb(image_path), image_size)
    np.save(cache_path, np.ascontiguousarray(resized, dtype=np.uint8), allow_pickle=False)
    meta_path.write_text(json.dumps(_build_metadata(image_path, image_size), ensure_ascii=False, indent=2), encoding="utf-8")
    return "updated" if existed else "created"


def ensure_resize_cache(
    image_paths: Iterable[str | Path],
    image_size: int,
    cache_dir: str | Path,
    num_workers: int = 4,
) -> dict[str, int]:
    unique_paths = sorted({str(Path(path).resolve()) for path in image_paths})
    stale_paths = [Path(path) for path in unique_paths if is_resize_cache_stale(cache_dir, path, image_size)]
    summary = {
        "total": len(unique_paths),
        "updated": 0,
        "skipped": len(unique_paths) - len(stale_paths),
    }
    if not stale_paths:
        return summary

    worker_count = max(1, min(int(num_workers), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(lambda path: _write_resize_cache(cache_dir, path, image_size), stale_paths))
    summary["updated"] = len(stale_paths)
    return summary
