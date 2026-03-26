from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_annotations(payload: dict, split_name: str) -> list[str]:
    errors: list[str] = []
    image_ids = {item["id"] for item in payload["images"]}
    category_ids = {item["id"] for item in payload["categories"]}
    for ann in payload["annotations"]:
        if ann["image_id"] not in image_ids:
            errors.append(f"{split_name}: annotation {ann['id']} references missing image_id {ann['image_id']}")
        if ann["category_id"] not in category_ids:
            errors.append(f"{split_name}: annotation {ann['id']} references missing category_id {ann['category_id']}")
        x1, y1, x2, y2 = ann["bbox"]
        if not (x2 > x1 and y2 > y1):
            errors.append(f"{split_name}: annotation {ann['id']} has invalid bbox {ann['bbox']}")
    return errors


def _validate_prompt_spec(prompt_spec: dict, train_payload: dict, prompt_spec_path: Path) -> list[str]:
    errors: list[str] = []
    image_by_name = {item["file_name"]: item for item in train_payload["images"]}
    anns_by_image = {}
    for ann in train_payload["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for prompt_idx, prompt in enumerate(prompt_spec.get("prompts", [])):
        image_path = Path(prompt["image"])
        file_name = image_path.name
        image = image_by_name.get(file_name)
        if image is None:
            errors.append(f"prompt[{prompt_idx}] references unknown image file {prompt['image']}")
            continue
        gt_anns = anns_by_image.get(image["id"], [])
        for ann_idx, ann in enumerate(prompt.get("annotations", [])):
            matched = any(
                gt_ann["category_id"] == ann["label"] and gt_ann["bbox"] == ann["bbox"]
                for gt_ann in gt_anns
            )
            if not matched:
                errors.append(
                    f"prompt[{prompt_idx}].annotations[{ann_idx}] "
                    f"does not match train.json for {file_name}: label={ann['label']} bbox={ann['bbox']}"
                )
        resolved = (prompt_spec_path.parent / prompt["image"]).resolve()
        if not resolved.exists():
            errors.append(f"prompt[{prompt_idx}] image path does not exist after resolution: {resolved}")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Validate toy_data annotations and prompt_set consistency.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to generated toy_data directory.")
    parser.add_argument("--prompt-spec", type=str, default=None, help="Path to prompt_set.json. Defaults to <data-dir>/prompt_set.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    prompt_spec_path = Path(args.prompt_spec) if args.prompt_spec else data_dir / "prompt_set.json"

    train_payload = _load_json(data_dir / "train.json")
    val_payload = _load_json(data_dir / "val.json")
    prompt_spec = _load_json(prompt_spec_path)

    errors = []
    errors.extend(_validate_annotations(train_payload, "train"))
    errors.extend(_validate_annotations(val_payload, "val"))
    errors.extend(_validate_prompt_spec(prompt_spec, train_payload, prompt_spec_path))

    if errors:
        print("CONSISTENCY_CHECK_FAILED")
        for error in errors:
            print(error)
        raise SystemExit(1)

    print("CONSISTENCY_CHECK_OK")
    print(f"train_images={len(train_payload['images'])} train_annotations={len(train_payload['annotations'])}")
    print(f"val_images={len(val_payload['images'])} val_annotations={len(val_payload['annotations'])}")
    print(f"prompt_images={len(prompt_spec.get('prompts', []))}")


if __name__ == "__main__":
    main()
