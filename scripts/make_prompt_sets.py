from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_annotations_for_labels(anns: list[dict], labels: list[int], rng: random.Random) -> list[dict]:
    selected = []
    for label in labels:
        candidates = [ann for ann in anns if ann["category_id"] == label]
        if not candidates:
            continue
        ann = rng.choice(candidates)
        selected.append({
            "bbox": ann["bbox"],
            "label": ann["category_id"],
        })
    return selected


def build_prompt_sets(
    annotations_path: Path,
    output_dir: Path,
    num_sets: int,
    min_prompt_classes: int,
    max_prompt_classes: int,
    max_extra_prompt_images: int,
    seed: int,
) -> list[dict]:
    payload = _load_json(annotations_path)
    rng = random.Random(seed)

    image_by_id = {item["id"]: item for item in payload["images"]}
    category_by_id = {item["id"]: item["name"] for item in payload["categories"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    class_to_image_ids: dict[int, set[int]] = defaultdict(set)
    class_image_to_anns: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for ann in payload["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
        class_to_image_ids[ann["category_id"]].add(ann["image_id"])
        class_image_to_anns[ann["category_id"]][ann["image_id"]].append(ann)

    candidate_image_ids = [
        image_id
        for image_id, anns in anns_by_image.items()
        if len({ann["category_id"] for ann in anns}) >= min_prompt_classes
    ]
    if not candidate_image_ids:
        raise ValueError("No images contain enough distinct categories to build prompt sets.")

    rng.shuffle(candidate_image_ids)
    selected_image_ids = candidate_image_ids[:num_sets]
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = annotations_path.parent / "images"
    relative_images_dir = Path(os.path.relpath(images_dir, output_dir))

    manifest = []
    for prompt_idx, image_id in enumerate(selected_image_ids):
        anns = anns_by_image[image_id]
        available_labels = sorted({ann["category_id"] for ann in anns})
        max_classes_for_image = min(len(available_labels), max_prompt_classes)
        prompt_class_count = rng.randint(min_prompt_classes, max_classes_for_image)
        selected_labels = sorted(rng.sample(available_labels, prompt_class_count))

        prompts = [{
            "image": str(relative_images_dir / image_by_id[image_id]["file_name"]),
            "annotations": _select_annotations_for_labels(anns, selected_labels, rng),
        }]
        used_image_ids = {image_id}

        extra_labels = selected_labels[:]
        rng.shuffle(extra_labels)
        for label in extra_labels:
            if len(prompts) - 1 >= max_extra_prompt_images:
                break
            alternative_image_ids = [alt_id for alt_id in sorted(class_to_image_ids[label]) if alt_id not in used_image_ids]
            if not alternative_image_ids:
                continue
            alt_image_id = rng.choice(alternative_image_ids)
            alt_ann = rng.choice(class_image_to_anns[label][alt_image_id])
            prompts.append({
                "image": str(relative_images_dir / image_by_id[alt_image_id]["file_name"]),
                "annotations": [{
                    "bbox": alt_ann["bbox"],
                    "label": alt_ann["category_id"],
                }],
            })
            used_image_ids.add(alt_image_id)

        prompt_payload = {"prompts": prompts}
        prompt_path = output_dir / f"prompt_set_{prompt_idx:03d}.json"
        prompt_path.write_text(json.dumps(prompt_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({
            "path": str(prompt_path),
            "source_image": image_by_id[image_id]["file_name"],
            "labels": selected_labels,
            "label_names": [category_by_id[label] for label in selected_labels],
            "num_prompt_images": len(prompts),
        })

    index_payload = {
        "annotations": str(annotations_path),
        "generated": len(manifest),
        "items": manifest,
    }
    (output_dir / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multiple prompt_set JSON files from a labeled split.")
    parser.add_argument("--annotations", type=str, required=True, help="COCO-like annotations JSON, e.g. ./toy_data/val.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write prompt_set_XXX.json files into")
    parser.add_argument("--num-sets", type=int, default=24)
    parser.add_argument("--min-prompt-classes", type=int, default=2)
    parser.add_argument("--max-prompt-classes", type=int, default=3)
    parser.add_argument("--max-extra-prompt-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_prompt_sets(
        annotations_path=Path(args.annotations),
        output_dir=Path(args.output_dir),
        num_sets=args.num_sets,
        min_prompt_classes=args.min_prompt_classes,
        max_prompt_classes=args.max_prompt_classes,
        max_extra_prompt_images=args.max_extra_prompt_images,
        seed=args.seed,
    )
    print(json.dumps({"generated": len(manifest), "output_dir": args.output_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
