from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promptdet.data.yolo_io import load_image_list, parse_yolo_label_file


def _load_config_payload(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_config_inputs(config_path: Path) -> tuple[Path, Path, dict[int, str], dict]:
    payload = _load_config_payload(config_path)
    data_cfg = payload["data"]
    images_list = Path(data_cfg["val_list"]).resolve()
    labels_dir = Path(data_cfg["labels_dir"]).resolve()
    class_names = {int(key): str(value) for key, value in data_cfg.get("class_names", {}).items()}
    return images_list, labels_dir, class_names, data_cfg


def _randint_clamped(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid random range: [{low}, {high}]")
    return rng.randint(low, high)


def _sample_annotations_for_labels(
    anns: list[dict],
    labels: list[int],
    rng: random.Random,
) -> list[dict]:
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


def _available_labels_for_image(anns: list[dict]) -> list[int]:
    return sorted({ann["category_id"] for ann in anns})


def _load_split_records(
    images_list_path: Path,
    labels_dir: Path,
    skip_missing_labels: bool,
) -> tuple[
    dict[int, dict],
    dict[int, list[dict]],
    dict[int, set[int]],
    dict[int, dict[int, list[dict]]],
    list[dict],
]:
    image_by_id: dict[int, dict] = {}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    class_to_image_ids: dict[int, set[int]] = defaultdict(set)
    class_image_to_anns: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped: list[dict] = []

    for image_id, image_path in enumerate(load_image_list(images_list_path)):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            if skip_missing_labels:
                skipped.append({
                    "image": str(image_path),
                    "reason": "missing_label",
                    "label_path": str(label_path),
                })
                continue
            raise FileNotFoundError(f"Missing label file for image {image_path}: {label_path}")

        items = parse_yolo_label_file(label_path)
        if not items:
            skipped.append({
                "image": str(image_path),
                "reason": "empty_label",
                "label_path": str(label_path),
            })
            continue

        image_by_id[image_id] = {
            "id": image_id,
            "path": image_path.resolve(),
            "file_name": image_path.name,
            "label_path": label_path.resolve(),
        }
        for category_id, bbox in items:
            ann = {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [float(value) for value in bbox],
            }
            anns_by_image[image_id].append(ann)
            class_to_image_ids[category_id].add(image_id)
            class_image_to_anns[category_id][image_id].append(ann)

    return image_by_id, anns_by_image, class_to_image_ids, class_image_to_anns, skipped


def build_prompt_specs(
    images_list_path: Path,
    labels_dir: Path,
    output_dir: Path,
    num_specs: int,
    min_prompt_classes: int,
    max_prompt_classes: int,
    min_prompt_images: int,
    max_prompt_images: int,
    class_names: dict[int, str] | None = None,
    seed: int = 42,
    skip_missing_labels: bool = True,
) -> dict:
    if num_specs <= 0:
        raise ValueError("num_specs must be positive.")
    if min_prompt_classes <= 0:
        raise ValueError("min_prompt_classes must be positive.")
    if max_prompt_classes < min_prompt_classes:
        raise ValueError("max_prompt_classes must be >= min_prompt_classes.")
    if min_prompt_images <= 0:
        raise ValueError("min_prompt_images must be positive.")
    if max_prompt_images < min_prompt_images:
        raise ValueError("max_prompt_images must be >= min_prompt_images.")

    rng = random.Random(seed)
    class_names = class_names or {}
    output_dir.mkdir(parents=True, exist_ok=True)

    image_by_id, anns_by_image, class_to_image_ids, class_image_to_anns, skipped = _load_split_records(
        images_list_path=images_list_path,
        labels_dir=labels_dir,
        skip_missing_labels=skip_missing_labels,
    )

    candidate_image_ids = [
        image_id
        for image_id, anns in anns_by_image.items()
        if len({ann["category_id"] for ann in anns}) >= min_prompt_classes
    ]
    if not candidate_image_ids:
        raise ValueError("No images contain enough distinct categories to build prompt specs.")

    items = []
    for spec_idx in range(num_specs):
        sampled = None
        max_attempts = 128
        for _ in range(max_attempts):
            source_image_id = rng.choice(candidate_image_ids)
            source_anns = anns_by_image[source_image_id]
            available_labels = _available_labels_for_image(source_anns)
            prompt_class_cap = min(max_prompt_classes, len(available_labels))
            prompt_class_count = _randint_clamped(rng, min_prompt_classes, prompt_class_cap)
            selected_labels = sorted(rng.sample(available_labels, prompt_class_count))

            candidate_extra_image_ids = sorted({
                image_id
                for label in selected_labels
                for image_id in class_to_image_ids[label]
                if image_id != source_image_id
            })
            prompt_image_cap = min(max_prompt_images, 1 + len(candidate_extra_image_ids))
            if prompt_image_cap < min_prompt_images:
                continue
            prompt_image_count = _randint_clamped(rng, min_prompt_images, prompt_image_cap)
            sampled = (source_image_id, source_anns, selected_labels, candidate_extra_image_ids, prompt_image_count)
            break

        if sampled is None:
            raise ValueError(
                "Failed to sample a prompt spec that satisfies the requested "
                f"min_prompt_images={min_prompt_images}. Try lowering min_prompt_images or max_prompt_classes."
            )

        source_image_id, source_anns, selected_labels, candidate_extra_image_ids, prompt_image_count = sampled

        prompts = [{
            "image": str(image_by_id[source_image_id]["path"]),
            "annotations": _sample_annotations_for_labels(source_anns, selected_labels, rng),
        }]
        used_image_ids = {source_image_id}
        target_extra_count = prompt_image_count - 1

        extra_image_ids = candidate_extra_image_ids[:]
        rng.shuffle(extra_image_ids)
        for alt_image_id in extra_image_ids:
            if len(prompts) - 1 >= target_extra_count:
                break
            if alt_image_id in used_image_ids:
                continue
            alt_available_labels = [
                label for label in selected_labels
                if class_image_to_anns[label].get(alt_image_id)
            ]
            if not alt_available_labels:
                continue
            prompts.append({
                "image": str(image_by_id[alt_image_id]["path"]),
                "annotations": _sample_annotations_for_labels(
                    anns_by_image[alt_image_id],
                    alt_available_labels,
                    rng,
                ),
            })
            used_image_ids.add(alt_image_id)

        prompt_payload = {"prompts": prompts}
        spec_path = output_dir / f"prompt_spec_{spec_idx:04d}.json"
        spec_path.write_text(json.dumps(prompt_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        items.append({
            "path": str(spec_path),
            "source_image": str(image_by_id[source_image_id]["path"]),
            "source_label_path": str(image_by_id[source_image_id]["label_path"]),
            "labels": selected_labels,
            "label_names": [class_names.get(label, str(label)) for label in selected_labels],
            "num_prompt_images": len(prompts),
        })

    manifest = {
        "images_list": str(images_list_path),
        "labels_dir": str(labels_dir),
        "generated": len(items),
        "num_specs": int(num_specs),
        "min_prompt_classes": int(min_prompt_classes),
        "max_prompt_classes": int(max_prompt_classes),
        "min_prompt_images": int(min_prompt_images),
        "max_prompt_images": int(max_prompt_images),
        "seed": int(seed),
        "skipped": skipped,
        "items": items,
    }
    (output_dir / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sampled prompt_spec JSON files from an image-list txt and YOLO labels."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON. Uses data.val_list, data.labels_dir, and prompt sampling defaults from data.",
    )
    parser.add_argument("--images-list", type=str, default=None, help="Path to txt file with one image path per line.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Directory containing YOLO label txt files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write prompt_spec JSON files into.")
    parser.add_argument("--num-specs", type=int, required=True, help="Number of prompt_spec JSON files to generate.")
    parser.add_argument("--min-prompt-classes", type=int, default=None, help="Minimum distinct labels in each prompt_spec.")
    parser.add_argument("--max-prompt-classes", type=int, default=None, help="Maximum distinct labels in each prompt_spec.")
    parser.add_argument("--min-prompt-images", type=int, default=1, help="Minimum prompt images in each prompt_spec.")
    parser.add_argument("--max-prompt-images", type=int, default=None, help="Maximum prompt images in each prompt_spec.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing-labels", action="store_true", help="Skip images whose label txt is missing.")
    return parser.parse_args()


def main():
    args = parse_args()

    config_class_names: dict[int, str] = {}
    data_cfg: dict = {}
    images_list_path: Path | None = None
    labels_dir: Path | None = None
    if args.config:
        config_path = Path(args.config).resolve()
        images_list_path, labels_dir, config_class_names, data_cfg = _resolve_config_inputs(config_path)

    if args.images_list is not None:
        images_list_path = Path(args.images_list).resolve()
    if args.labels_dir is not None:
        labels_dir = Path(args.labels_dir).resolve()

    if images_list_path is None or labels_dir is None:
        raise ValueError("Provide either --config or both --images-list and --labels-dir.")

    min_prompt_classes = args.min_prompt_classes
    if min_prompt_classes is None:
        min_prompt_classes = int(data_cfg.get("min_prompt_classes", 1))

    max_prompt_classes = args.max_prompt_classes
    if max_prompt_classes is None:
        max_prompt_classes = int(data_cfg.get("max_prompt_classes", min_prompt_classes))

    max_prompt_images = args.max_prompt_images
    if max_prompt_images is None:
        max_prompt_images = int(data_cfg.get("max_prompt_images", args.min_prompt_images))

    manifest = build_prompt_specs(
        images_list_path=images_list_path,
        labels_dir=labels_dir,
        output_dir=Path(args.output_dir).resolve(),
        num_specs=args.num_specs,
        min_prompt_classes=int(min_prompt_classes),
        max_prompt_classes=int(max_prompt_classes),
        min_prompt_images=int(args.min_prompt_images),
        max_prompt_images=int(max_prompt_images),
        class_names=config_class_names,
        seed=int(args.seed),
        skip_missing_labels=args.skip_missing_labels or args.config is not None,
    )
    print(json.dumps({
        "generated": manifest["generated"],
        "skipped": len(manifest["skipped"]),
        "output_dir": str(Path(args.output_dir).resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
