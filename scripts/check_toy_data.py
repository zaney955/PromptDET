from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promptdet.data.yolo_io import load_class_names, load_image_list, parse_yolo_label_file


def _valid_yolo_box(box: list[float]) -> bool:
    if len(box) != 4:
        return False
    cx, cy, w, h = [float(value) for value in box]
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_split(
    data_dir: Path,
    split_name: str,
    class_names: dict[int, str],
) -> tuple[list[str], dict[str, list[tuple[int, list[float]]]], int, int]:
    errors: list[str] = []
    image_list_path = data_dir / f"{split_name}.txt"
    labels_dir = data_dir / "labels" / split_name
    if not image_list_path.exists():
        return [f"{split_name}: missing image list {image_list_path}"], {}, 0, 0
    if not labels_dir.exists():
        return [f"{split_name}: missing labels directory {labels_dir}"], {}, 0, 0

    image_paths = load_image_list(image_list_path)
    anns_by_file: dict[str, list[tuple[int, list[float]]]] = {}
    label_stems = {
        path.stem
        for path in labels_dir.glob("*.txt")
        if not path.name.startswith("._")
    }
    used_stems: set[str] = set()
    annotation_count = 0

    for image_path in image_paths:
        if not image_path.exists():
            errors.append(f"{split_name}: missing image file {image_path}")
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"{split_name}: missing label file {label_path}")
            continue
        rows = parse_yolo_label_file(label_path)
        used_stems.add(image_path.stem)
        anns_by_file[image_path.name] = rows
        annotation_count += len(rows)
        for row_idx, (class_id, bbox) in enumerate(rows):
            if class_id not in class_names:
                errors.append(f"{split_name}: {label_path} line {row_idx + 1} uses unknown class_id {class_id}")
            if not _valid_yolo_box(bbox):
                errors.append(f"{split_name}: {label_path} line {row_idx + 1} has invalid bbox {bbox}")

    extra_stems = sorted(label_stems - used_stems)
    for stem in extra_stems:
        errors.append(f"{split_name}: extra label file without image list entry {labels_dir / f'{stem}.txt'}")

    return errors, anns_by_file, len(image_paths), annotation_count


def _validate_prompt_spec(
    prompt_spec_path: Path,
    train_annotations: dict[str, list[tuple[int, list[float]]]],
    class_names: dict[int, str],
) -> list[str]:
    errors: list[str] = []
    prompt_spec = _load_json(prompt_spec_path)

    for prompt_idx, prompt in enumerate(prompt_spec.get("prompts", [])):
        resolved = (prompt_spec_path.parent / prompt["image"]).resolve()
        if not resolved.exists():
            errors.append(f"prompt[{prompt_idx}] image path does not exist after resolution: {resolved}")
            continue
        gt_anns = train_annotations.get(resolved.name)
        if gt_anns is None:
            errors.append(f"prompt[{prompt_idx}] references image not present in train split: {resolved.name}")
            continue
        for ann_idx, ann in enumerate(prompt.get("annotations", [])):
            label = int(ann["label"])
            bbox = ann["bbox"]
            if label not in class_names:
                errors.append(f"prompt[{prompt_idx}].annotations[{ann_idx}] uses unknown label {label}")
                continue
            if not _valid_yolo_box(bbox):
                errors.append(f"prompt[{prompt_idx}].annotations[{ann_idx}] has invalid bbox {bbox}")
                continue
            if not any(gt_label == label and gt_bbox == bbox for gt_label, gt_bbox in gt_anns):
                errors.append(
                    f"prompt[{prompt_idx}].annotations[{ann_idx}] "
                    f"does not match train labels for {resolved.name}: label={label} bbox={bbox}"
                )
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Validate toy_data labels, split files, and prompt_set consistency.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to generated toy_data directory.")
    parser.add_argument("--prompt-spec", type=str, default=None, help="Path to prompt_set.json. Defaults to <data-dir>/prompt_set.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    prompt_spec_path = Path(args.prompt_spec) if args.prompt_spec else data_dir / "prompt_set.json"
    class_names_path = data_dir / "classes.txt"
    if not class_names_path.exists():
        raise SystemExit(f"Missing classes file: {class_names_path}")

    class_names = load_class_names(class_names_path)
    train_errors, train_annotations, train_images, train_annotation_count = _validate_split(data_dir, "train", class_names)
    val_errors, _, val_images, val_annotation_count = _validate_split(data_dir, "val", class_names)
    prompt_errors = _validate_prompt_spec(prompt_spec_path, train_annotations, class_names)
    errors = [*train_errors, *val_errors, *prompt_errors]

    if errors:
        print("CONSISTENCY_CHECK_FAILED")
        for error in errors:
            print(error)
        raise SystemExit(1)

    prompt_spec = _load_json(prompt_spec_path)
    print("CONSISTENCY_CHECK_OK")
    print(f"train_images={train_images} train_annotations={train_annotation_count}")
    print(f"val_images={val_images} val_annotations={val_annotation_count}")
    print(f"prompt_images={len(prompt_spec.get('prompts', []))}")


if __name__ == "__main__":
    main()
