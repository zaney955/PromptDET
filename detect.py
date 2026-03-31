from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from promptdet.config import load_config
from promptdet.data.prompt_hints import (
    build_prompt_hint_map,
    build_prompt_target_map,
    sample_slot_colors,
)
from promptdet.models.promptdet import PromptDET
from promptdet.utils.box_formats import yolo_xywh_to_xyxy_tensor
from promptdet.utils.checkpoint import load_checkpoint
from promptdet.utils.visualize import save_detection_visualization, save_grounding_visualizations


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def resize_image(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return pil_to_tensor(image)


def _valid_yolo_box(box: list[float] | tuple[float, ...]) -> bool:
    if len(box) != 4:
        return False
    center_x, center_y, width, height = [float(value) for value in box]
    return 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run PromptDET inference on one prompt-query pair or a prompt set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt-spec", type=str, default=None, help="JSON file describing a prompt set.")
    parser.add_argument("--prompt-image", type=str, default=None)
    parser.add_argument("--prompt-box", type=float, nargs=4, default=None, help="normalized x_center y_center width height on the original prompt image")
    parser.add_argument("--prompt-label", type=int, default=None)
    parser.add_argument("--query-image", type=str, required=True, help="Path to one query image or a directory of query images.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--pre-score-topk", type=int, default=None)
    parser.add_argument("--local-peak-kernel", type=int, default=None)
    parser.add_argument("--oversize-box-threshold", type=float, default=None)
    parser.add_argument("--oversize-box-gamma", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    return parser.parse_args()


def _iter_query_images(query_path: Path) -> list[Path]:
    if query_path.is_file():
        return [query_path]
    if not query_path.is_dir():
        raise ValueError(f"--query-image must be an image file or directory, got: {query_path}")
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(path for path in query_path.iterdir() if path.is_file() and path.suffix.lower() in image_suffixes)
    if not image_paths:
        raise ValueError(f"No query images found in directory: {query_path}")
    return image_paths


def _load_prompt_set(
    args,
    image_size: int,
    max_prompt_classes: int,
    hint_inner_shrink: float,
    hint_bg_expand: float,
    random_color_min_distance: float,
    center_target_sigma: float,
):
    if args.prompt_spec:
        prompt_spec_path = Path(args.prompt_spec).resolve()
        payload = json.loads(prompt_spec_path.read_text(encoding="utf-8"))
        prompts = payload["prompts"]
    else:
        prompt_spec_path = None
        if args.prompt_image is None or args.prompt_box is None or args.prompt_label is None:
            raise ValueError("Single prompt mode requires --prompt-image, --prompt-box and --prompt-label.")
        prompts = [{
            "image": args.prompt_image,
            "annotations": [{"bbox": args.prompt_box, "label": args.prompt_label}],
        }]

    prompt_images = []
    prompt_boxes = []
    prompt_hint_maps = []
    prompt_target_maps = []
    prompt_class_indices = []
    label_to_slot = {}

    for prompt in prompts:
        prompt_image_path = Path(prompt["image"])
        if not prompt_image_path.is_absolute() and prompt_spec_path is not None:
            prompt_image_path = (prompt_spec_path.parent / prompt_image_path).resolve()
        prompt_image = Image.open(prompt_image_path).convert("RGB")
        resized = resize_image(prompt_image, image_size)
        for ann in prompt["annotations"]:
            label = int(ann["label"])
            bbox_raw = ann["bbox"]
            if not _valid_yolo_box(bbox_raw):
                raise ValueError(
                    "prompt_spec annotations must use normalized YOLO xywh boxes in [0, 1]. "
                    f"Got bbox={bbox_raw} for image={prompt_image_path} label={label}."
                )
            slot = label_to_slot.setdefault(label, len(label_to_slot))
            bbox = yolo_xywh_to_xyxy_tensor(
                torch.tensor([bbox_raw], dtype=torch.float32),
                prompt_image.size[0],
                prompt_image.size[1],
            )[0]
            bbox[0::2] *= image_size / prompt_image.size[0]
            bbox[1::2] *= image_size / prompt_image.size[1]
            prompt_images.append(resized)
            prompt_boxes.append(bbox)
            hint = build_prompt_hint_map(
                image_size,
                bbox,
                inner_shrink=hint_inner_shrink,
                bg_expand=hint_bg_expand,
            )
            prompt_hint_maps.append(hint)
            prompt_class_indices.append(slot)

    if not prompt_images:
        raise ValueError("Prompt set is empty.")
    if len(label_to_slot) > max_prompt_classes:
        raise ValueError("Prompt set exceeds model.max_prompt_classes.")

    prompt_class_ids = [None] * len(label_to_slot)
    for label, slot in label_to_slot.items():
        prompt_class_ids[slot] = label
    slot_colors = sample_slot_colors(len(label_to_slot), min_distance=random_color_min_distance)
    prompt_target_maps = [
        build_prompt_target_map(
            image_size,
            box,
            int(class_idx),
            slot_colors,
            center_sigma=center_target_sigma,
        )
        for box, class_idx in zip(prompt_boxes, prompt_class_indices)
    ]

    return {
        "prompt_images": torch.stack(prompt_images, dim=0).unsqueeze(0),
        "prompt_boxes": torch.stack(prompt_boxes, dim=0).unsqueeze(0),
        "prompt_hint_maps": torch.stack(prompt_hint_maps, dim=0).unsqueeze(0),
        "prompt_target_maps": torch.stack(prompt_target_maps, dim=0).unsqueeze(0),
        "prompt_class_indices": torch.tensor(prompt_class_indices, dtype=torch.long).unsqueeze(0),
        "prompt_instance_mask": torch.ones((1, len(prompt_images)), dtype=torch.bool),
        "prompt_class_ids": torch.tensor(prompt_class_ids, dtype=torch.long).unsqueeze(0),
        "prompt_class_mask": torch.ones((1, len(prompt_class_ids)), dtype=torch.bool),
        "slot_colors": slot_colors.unsqueeze(0),
    }


def _run_single_query(
    model: PromptDET,
    prompt_batch: dict[str, torch.Tensor],
    query_path: Path,
    device: torch.device,
    image_size: int,
    score_threshold: float,
    pre_score_topk: int,
    local_peak_kernel: int,
    oversize_box_threshold: float,
    oversize_box_gamma: float,
    max_detections: int,
) -> tuple[Image.Image, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    query_pil = Image.open(query_path).convert("RGB")
    query_tensor = resize_image(query_pil, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(
            prompt_batch["prompt_images"].to(device),
            prompt_batch["prompt_boxes"].to(device),
            prompt_batch["prompt_hint_maps"].to(device),
            prompt_batch["prompt_target_maps"].to(device),
            prompt_batch["prompt_class_indices"].to(device),
            prompt_batch["prompt_instance_mask"].to(device),
            prompt_batch["prompt_class_mask"].to(device),
            query_tensor,
        )
        preds = model.predict(
            raw,
            prompt_batch["prompt_class_ids"].to(device),
            prompt_batch["prompt_class_mask"].to(device),
            image_size=image_size,
            score_threshold=score_threshold,
            pre_score_topk=pre_score_topk,
            local_peak_kernel=local_peak_kernel,
            oversize_box_threshold=oversize_box_threshold,
            oversize_box_gamma=oversize_box_gamma,
            max_detections=max_detections,
        )[0]

    boxes = preds["boxes"].cpu()
    boxes[:, 0::2] *= query_pil.size[0] / image_size
    boxes[:, 1::2] *= query_pil.size[1] / image_size
    preds["boxes"] = boxes
    return query_pil, preds, raw.get("context_aux")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.train.device = args.device
    if args.score_threshold is not None:
        config.train.score_threshold = args.score_threshold
    if args.pre_score_topk is not None:
        config.train.pre_score_topk = args.pre_score_topk
    if args.local_peak_kernel is not None:
        config.train.local_peak_kernel = args.local_peak_kernel
    if args.oversize_box_threshold is not None:
        config.train.oversize_box_threshold = args.oversize_box_threshold
    if args.oversize_box_gamma is not None:
        config.train.oversize_box_gamma = args.oversize_box_gamma
    if args.max_detections is not None:
        config.train.max_detections = args.max_detections
    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")

    model = PromptDET(config.model, config.dense_grounding).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device.type)
    model.eval()
    model.set_context_prior_strength(1.0)

    prompt_batch = _load_prompt_set(
        args,
        config.model.image_size,
        config.model.max_prompt_classes,
        config.dense_grounding.hint_inner_shrink,
        config.dense_grounding.hint_bg_expand,
        config.dense_grounding.random_color_min_distance,
        config.loss.center_target_sigma,
    )
    query_path = Path(args.query_image).resolve()
    query_images = _iter_query_images(query_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_mode = query_path.is_dir()
    batch_summary = []

    for current_query_path in query_images:
        query_pil, preds, grounding_aux = _run_single_query(
            model,
            prompt_batch,
            current_query_path,
            device=device,
            image_size=config.model.image_size,
            score_threshold=config.train.score_threshold,
            pre_score_topk=config.train.pre_score_topk,
            local_peak_kernel=config.train.local_peak_kernel,
            oversize_box_threshold=config.train.oversize_box_threshold,
            oversize_box_gamma=config.train.oversize_box_gamma,
            max_detections=config.train.max_detections,
        )
        payload = {
            "boxes": preds["boxes"].tolist(),
            "scores": preds["scores"].cpu().tolist(),
            "labels": preds["labels"].cpu().tolist(),
        }
        current_output_dir = output_dir / current_query_path.stem if batch_mode else output_dir
        current_output_dir.mkdir(parents=True, exist_ok=True)
        save_detection_visualization(query_pil, preds, current_output_dir / "prediction.png")
        (current_output_dir / "prediction.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if grounding_aux is not None:
            save_grounding_visualizations(
                grounding_aux,
                current_output_dir,
                prompt_targets=prompt_batch["prompt_target_maps"][0],
            )
            (current_output_dir / "grounding_debug.json").write_text(
                json.dumps(
                    {
                        "query_image": str(current_query_path),
                        "num_prompts": int(prompt_batch["prompt_images"].shape[1]),
                        "slot_colors": prompt_batch["slot_colors"][0].tolist(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if batch_mode:
            batch_summary.append({
                "image": str(current_query_path),
                "output_dir": str(current_output_dir),
                "num_detections": len(payload["boxes"]),
            })
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

    if batch_mode:
        summary_path = output_dir / "batch_summary.json"
        summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(batch_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
