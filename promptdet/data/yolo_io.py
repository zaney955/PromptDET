from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_class_names(path: str | Path) -> dict[int, str]:
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {idx: name for idx, name in enumerate(names)}


def load_image_list(path: str | Path) -> list[Path]:
    list_path = Path(path)
    image_paths: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        image_path = Path(item)
        if not image_path.is_absolute():
            image_path = (list_path.parent / image_path).resolve()
        image_paths.append(image_path)
    return image_paths


def imread_rgb(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to decode image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def probe_image_size(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    with image_path.open("rb") as handle:
        header = handle.read(64 * 1024)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    if header.startswith(b"\xff\xd8"):
        idx = 2
        while idx + 9 < len(header):
            if header[idx] != 0xFF:
                idx += 1
                continue
            marker = header[idx + 1]
            idx += 2
            if marker in {0xD8, 0xD9}:
                continue
            if idx + 2 > len(header):
                break
            block_len = int.from_bytes(header[idx:idx + 2], "big")
            if block_len < 2 or idx + block_len > len(header):
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height = int.from_bytes(header[idx + 3:idx + 5], "big")
                width = int.from_bytes(header[idx + 5:idx + 7], "big")
                return width, height
            idx += block_len
    image = imread_rgb(image_path)
    height, width = image.shape[:2]
    return width, height


def parse_yolo_label_file(path: str | Path) -> list[tuple[int, list[float]]]:
    label_path = Path(path)
    if not label_path.exists():
        return []
    items: list[tuple[int, list[float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line!r}")
        class_id = int(parts[0])
        bbox = [float(value) for value in parts[1:]]
        items.append((class_id, bbox))
    return items
