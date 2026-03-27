from __future__ import annotations

from pathlib import Path


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
