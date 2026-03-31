from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promptdet.utils.box_formats import xyxy_to_yolo_xywh


CLASSES = [
    "square",
    "circle",
    "triangle",
    "diamond",
    "pentagon",
    "hexagon",
    "star",
    "cross",
]

COLOR_PALETTE = [
    (230, 70, 70),
    (70, 120, 230),
    (70, 190, 90),
    (230, 200, 70),
    (170, 90, 210),
    (70, 200, 210),
    (230, 150, 80),
    (220, 110, 170),
    (120, 120, 120),
    (120, 70, 30),
    (40, 160, 160),
    (180, 80, 120),
]


def _regular_polygon_points(x1: float, y1: float, x2: float, y2: float, sides: int, rotation: float = 0.0):
    import math

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    radius = min(x2 - x1, y2 - y1) / 2
    points = []
    for idx in range(sides):
        angle = rotation + (2.0 * math.pi * idx / sides)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _star_points(x1: float, y1: float, x2: float, y2: float, points_count: int = 5):
    import math

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    outer_radius = min(x2 - x1, y2 - y1) / 2
    inner_radius = outer_radius * 0.45
    points = []
    for idx in range(points_count * 2):
        angle = -math.pi / 2 + idx * math.pi / points_count
        radius = outer_radius if idx % 2 == 0 else inner_radius
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def draw_shape(draw: ImageDraw.ImageDraw, bbox, color, shape):
    x1, y1, x2, y2 = bbox
    if shape == "square":
        draw.rectangle([x1, y1, x2, y2], fill=color)
        return
    if shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)
        return
    if shape == "triangle":
        points = [(x1 + (x2 - x1) / 2, y1), (x2, y2), (x1, y2)]
        draw.polygon(points, fill=color)
        return
    if shape == "diamond":
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        draw.polygon(points, fill=color)
        return
    if shape == "pentagon":
        draw.polygon(_regular_polygon_points(x1, y1, x2, y2, sides=5, rotation=-1.57079632679), fill=color)
        return
    if shape == "hexagon":
        draw.polygon(_regular_polygon_points(x1, y1, x2, y2, sides=6, rotation=-1.57079632679), fill=color)
        return
    if shape == "star":
        draw.polygon(_star_points(x1, y1, x2, y2, points_count=5), fill=color)
        return
    if shape == "cross":
        width = x2 - x1
        height = y2 - y1
        arm_w = width * 0.3
        arm_h = height * 0.3
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = [
            (cx - arm_w / 2, y1),
            (cx + arm_w / 2, y1),
            (cx + arm_w / 2, cy - arm_h / 2),
            (x2, cy - arm_h / 2),
            (x2, cy + arm_h / 2),
            (cx + arm_w / 2, cy + arm_h / 2),
            (cx + arm_w / 2, y2),
            (cx - arm_w / 2, y2),
            (cx - arm_w / 2, cy + arm_h / 2),
            (x1, cy + arm_h / 2),
            (x1, cy - arm_h / 2),
            (cx - arm_w / 2, cy - arm_h / 2),
        ]
        draw.polygon(points, fill=color)
        return
    raise ValueError(f"Unsupported shape: {shape}")


def sample_bbox(image_size: int, min_size: int = 28, max_size: int = 96):
    size = random.randint(min_size, max_size)
    x1 = random.randint(0, image_size - size - 1)
    y1 = random.randint(0, image_size - size - 1)
    return [x1, y1, x1 + size, y1 + size]


def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(value), 6) for value in bbox]


def _write_split_list(output_dir: Path, split: str, file_names: list[str]) -> None:
    list_rows = [f"./images/{file_name}" for file_name in file_names]
    (output_dir / f"{split}.txt").write_text("\n".join(list_rows) + "\n", encoding="utf-8")


def _write_classes_file(output_dir: Path) -> None:
    (output_dir / "classes.txt").write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8",
    )


def _write_dataset_yaml(output_dir: Path) -> None:
    names_rows = [f"  {idx}: {name}" for idx, name in enumerate(CLASSES)]
    payload = [
        "path: .",
        "train: train.txt",
        "val: val.txt",
        "names:",
        *names_rows,
        "",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(payload), encoding="utf-8")


def build_split(output_dir: Path, split: str, num_images: int, image_size: int) -> list[dict]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    file_names: list[str] = []
    for image_id in range(num_images):
        image = Image.new("RGB", (image_size, image_size), color=(248, 248, 248))
        draw = ImageDraw.Draw(image)
        num_objects = random.randint(2, 6)
        label_rows = []
        annotations = []
        for _ in range(num_objects):
            category_id = random.randint(0, len(CLASSES) - 1)
            shape = CLASSES[category_id]
            color = random.choice(COLOR_PALETTE)
            bbox_xyxy = sample_bbox(image_size)
            draw_shape(draw, bbox_xyxy, color, shape)
            bbox_yolo = _round_bbox(xyxy_to_yolo_xywh(bbox_xyxy, image_size, image_size))
            annotations.append({
                "category_id": category_id,
                "bbox": bbox_yolo,
            })
            label_rows.append(
                f"{category_id} "
                f"{bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}"
            )
        file_name = f"{split}_{image_id:05d}.png"
        image.save(images_dir / file_name)
        (labels_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(label_rows) + "\n", encoding="utf-8")
        file_names.append(file_name)
        records.append({
            "id": image_id,
            "file_name": file_name,
            "image": f"./images/{file_name}",
            "annotations": annotations,
        })

    _write_split_list(output_dir, split, file_names)
    return records


def build_prompt_spec(output_dir: Path, train_records: list[dict]):
    anns_by_image = defaultdict(list)
    for record in train_records:
        anns_by_image[record["id"]].extend(record["annotations"])

    image_ids = sorted(anns_by_image.keys())
    if not image_ids:
        raise ValueError("train split has no annotations; cannot build prompt_set.json")

    primary_image_id = image_ids[0]
    for image_id in image_ids:
        distinct_categories = {ann["category_id"] for ann in anns_by_image[image_id]}
        if len(distinct_categories) >= 2:
            primary_image_id = image_id
            break

    record_by_id = {record["id"]: record for record in train_records}
    selected_annotations = []
    selected_labels = []
    seen_labels = set()
    for ann in anns_by_image[primary_image_id]:
        if ann["category_id"] in seen_labels:
            continue
        selected_annotations.append({
            "bbox": ann["bbox"],
            "label": ann["category_id"],
        })
        selected_labels.append(ann["category_id"])
        seen_labels.add(ann["category_id"])
        if len(selected_annotations) >= 2:
            break

    prompts = [{
        "image": record_by_id[primary_image_id]["image"],
        "annotations": selected_annotations,
    }]

    for label in selected_labels:
        found = None
        for image_id in image_ids:
            if image_id == primary_image_id:
                continue
            match = next((ann for ann in anns_by_image[image_id] if ann["category_id"] == label), None)
            if match is not None:
                found = {
                    "image": record_by_id[image_id]["image"],
                    "annotations": [{
                        "bbox": match["bbox"],
                        "label": match["category_id"],
                    }],
                }
                break
        if found is not None:
            prompts.append(found)
            break

    prompt_payload = {"prompts": prompts}
    (output_dir / "prompt_set.json").write_text(
        json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prompt_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Generate toy dataset for PromptDET.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-images", type=int, default=800)
    parser.add_argument("--val-images", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_classes_file(output_dir)
    _write_dataset_yaml(output_dir)
    train_records = build_split(output_dir, "train", args.train_images, args.image_size)
    build_split(output_dir, "val", args.val_images, args.image_size)
    build_prompt_spec(output_dir, train_records)
    print(f"Toy dataset generated at {output_dir}")


if __name__ == "__main__":
    main()
