from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from promptdet.config import load_config
from promptdet.models.promptdet import PromptDET
from promptdet.utils.checkpoint import load_checkpoint
from promptdet.utils.visualize import save_detection_visualization


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def resize_image(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return pil_to_tensor(image)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PromptDET inference on one prompt-query pair or a prompt set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt-spec", type=str, default=None, help="JSON file describing a prompt set.")
    parser.add_argument("--prompt-image", type=str, default=None)
    parser.add_argument("--prompt-box", type=float, nargs=4, default=None, help="x1 y1 x2 y2 on the original prompt image")
    parser.add_argument("--prompt-label", type=int, default=None)
    parser.add_argument("--query-image", type=str, required=True, help="Path to one query image or a directory of query images.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--pre-nms-topk", type=int, default=None)
    parser.add_argument("--one2one-topk", type=int, default=None)
    parser.add_argument("--one2one-peak-kernel", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
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


def _load_prompt_set(args, image_size: int, max_prompt_classes: int):
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
    prompt_class_indices = []
    label_to_slot = {}

    for prompt in prompts:
        prompt_image_path = Path(prompt["image"])
        if not prompt_image_path.is_absolute() and prompt_spec_path is not None:
            prompt_image_path = (prompt_spec_path.parent / prompt_image_path).resolve()
        prompt_image = Image.open(prompt_image_path).convert("RGB")
        resized = resize_image(prompt_image, image_size)
        scale_x = image_size / prompt_image.size[0]
        scale_y = image_size / prompt_image.size[1]
        for ann in prompt["annotations"]:
            label = int(ann["label"])
            slot = label_to_slot.setdefault(label, len(label_to_slot))
            bbox = torch.tensor(ann["bbox"], dtype=torch.float32)
            bbox[0::2] *= scale_x
            bbox[1::2] *= scale_y
            prompt_images.append(resized)
            prompt_boxes.append(bbox)
            prompt_class_indices.append(slot)

    if not prompt_images:
        raise ValueError("Prompt set is empty.")
    if len(label_to_slot) > max_prompt_classes:
        raise ValueError("Prompt set exceeds model.max_prompt_classes.")

    prompt_class_ids = [None] * len(label_to_slot)
    for label, slot in label_to_slot.items():
        prompt_class_ids[slot] = label

    return {
        "prompt_images": torch.stack(prompt_images, dim=0).unsqueeze(0),
        "prompt_boxes": torch.stack(prompt_boxes, dim=0).unsqueeze(0),
        "prompt_class_indices": torch.tensor(prompt_class_indices, dtype=torch.long).unsqueeze(0),
        "prompt_instance_mask": torch.ones((1, len(prompt_images)), dtype=torch.bool),
        "prompt_class_ids": torch.tensor(prompt_class_ids, dtype=torch.long).unsqueeze(0),
        "prompt_class_mask": torch.ones((1, len(prompt_class_ids)), dtype=torch.bool),
        "prompt_type": torch.tensor([0], dtype=torch.long),
    }


def _run_single_query(
    model: PromptDET,
    prompt_batch: dict[str, torch.Tensor],
    query_path: Path,
    device: torch.device,
    image_size: int,
    conf_threshold: float,
    nms_iou_threshold: float,
    pre_nms_topk: int,
    one2one_topk: int,
    one2one_peak_kernel: int,
    max_det: int,
) -> tuple[Image.Image, dict[str, torch.Tensor]]:
    query_pil = Image.open(query_path).convert("RGB")
    query_tensor = resize_image(query_pil, image_size).unsqueeze(0).to(device)
    if device.type == "cuda":
        query_tensor = query_tensor.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        raw = model(
            prompt_batch["prompt_images"].to(device),
            prompt_batch["prompt_boxes"].to(device),
            prompt_batch["prompt_class_indices"].to(device),
            prompt_batch["prompt_instance_mask"].to(device),
            prompt_batch["prompt_class_mask"].to(device),
            query_tensor,
            prompt_batch["prompt_type"].to(device),
        )
        preds = model.predict(
            raw,
            prompt_batch["prompt_class_ids"].to(device),
            prompt_batch["prompt_class_mask"].to(device),
            image_size=image_size,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            pre_nms_topk=pre_nms_topk,
            one2one_topk=one2one_topk,
            one2one_peak_kernel=one2one_peak_kernel,
            max_det=max_det,
        )[0]

    boxes = preds["boxes"].cpu()
    boxes[:, 0::2] *= query_pil.size[0] / image_size
    boxes[:, 1::2] *= query_pil.size[1] / image_size
    preds["boxes"] = boxes
    return query_pil, preds


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.train.device = args.device
    if args.conf_threshold is not None:
        config.train.conf_threshold = args.conf_threshold
    if args.nms_iou_threshold is not None:
        config.train.nms_iou_threshold = args.nms_iou_threshold
    if args.pre_nms_topk is not None:
        config.train.pre_nms_topk = args.pre_nms_topk
    if args.one2one_topk is not None:
        config.train.one2one_topk = args.one2one_topk
    if args.one2one_peak_kernel is not None:
        config.train.one2one_peak_kernel = args.one2one_peak_kernel
    if args.max_det is not None:
        config.train.max_det = args.max_det
    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")

    model = PromptDET(config.model).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    load_checkpoint(args.checkpoint, model, map_location=device.type)
    model.eval()

    prompt_batch = _load_prompt_set(args, config.model.image_size, config.model.max_prompt_classes)
    query_path = Path(args.query_image).resolve()
    query_images = _iter_query_images(query_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_mode = query_path.is_dir()
    batch_summary = []

    for current_query_path in query_images:
        query_pil, preds = _run_single_query(
            model,
            prompt_batch,
            current_query_path,
            device=device,
            image_size=config.model.image_size,
            conf_threshold=config.train.conf_threshold,
            nms_iou_threshold=config.train.nms_iou_threshold,
            pre_nms_topk=config.train.pre_nms_topk,
            one2one_topk=config.train.one2one_topk,
            one2one_peak_kernel=config.train.one2one_peak_kernel,
            max_det=config.train.max_det,
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
