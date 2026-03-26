from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = [
    ("red_square", (230, 70, 70), "square"),
    ("blue_square", (70, 120, 230), "square"),
    ("green_circle", (70, 190, 90), "circle"),
    ("yellow_circle", (230, 200, 70), "circle"),
    ("purple_triangle", (170, 90, 210), "triangle"),
    ("cyan_triangle", (70, 200, 210), "triangle"),
    ("orange_diamond", (230, 150, 80), "diamond"),
    ("pink_diamond", (220, 110, 170), "diamond"),
]


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


def sample_bbox(image_size: int, min_size: int = 28, max_size: int = 96):
    size = random.randint(min_size, max_size)
    x1 = random.randint(0, image_size - size - 1)
    y1 = random.randint(0, image_size - size - 1)
    return [x1, y1, x1 + size, y1 + size]


def build_split(output_dir: Path, split: str, num_images: int, image_size: int):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    payload = {"images": [], "annotations": [], "categories": []}
    for idx, (name, _, _) in enumerate(CLASSES):
        payload["categories"].append({"id": idx, "name": name})

    ann_id = 0
    for image_id in range(num_images):
        image = Image.new("RGB", (image_size, image_size), color=(248, 248, 248))
        draw = ImageDraw.Draw(image)
        num_objects = random.randint(2, 6)
        for _ in range(num_objects):
            category_id = random.randint(0, len(CLASSES) - 1)
            _, color, shape = CLASSES[category_id]
            bbox = sample_bbox(image_size)
            draw_shape(draw, bbox, color, shape)
            payload["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
            })
            ann_id += 1
        file_name = f"{split}_{image_id:05d}.png"
        image.save(images_dir / file_name)
        payload["images"].append({"id": image_id, "file_name": file_name, "width": image_size, "height": image_size})

    (output_dir / f"{split}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_prompt_spec(output_dir: Path, train_payload: dict):
    image_by_id = {item["id"]: item for item in train_payload["images"]}
    anns_by_image = defaultdict(list)
    for ann in train_payload["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = sorted(anns_by_image.keys())
    if not image_ids:
        raise ValueError("train split has no annotations; cannot build prompt_set.json")

    primary_image_id = image_ids[0]
    for image_id in image_ids:
        distinct_categories = {ann["category_id"] for ann in anns_by_image[image_id]}
        if len(distinct_categories) >= 2:
            primary_image_id = image_id
            break

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
        "image": f"./images/{image_by_id[primary_image_id]['file_name']}",
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
                    "image": f"./images/{image_by_id[image_id]['file_name']}",
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
    train_payload = build_split(output_dir, "train", args.train_images, args.image_size)
    build_split(output_dir, "val", args.val_images, args.image_size)
    build_prompt_spec(output_dir, train_payload)
    print(f"Toy dataset generated at {output_dir}")


if __name__ == "__main__":
    main()
