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
    parser = argparse.ArgumentParser(description="Run PromptDET inference on one prompt-query pair.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt-image", type=str, required=True)
    parser.add_argument("--prompt-box", type=float, nargs=4, required=True, help="x1 y1 x2 y2 on the original prompt image")
    parser.add_argument("--prompt-label", type=int, required=True)
    parser.add_argument("--query-image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.train.device = args.device
    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")

    model = PromptDET(config.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device.type)
    model.eval()

    prompt_pil = Image.open(args.prompt_image).convert("RGB")
    query_pil = Image.open(args.query_image).convert("RGB")
    prompt_tensor = resize_image(prompt_pil, config.model.image_size).unsqueeze(0).to(device)
    query_tensor = resize_image(query_pil, config.model.image_size).unsqueeze(0).to(device)

    box = torch.tensor([args.prompt_box], dtype=torch.float32)
    box[:, 0::2] *= config.model.image_size / prompt_pil.size[0]
    box[:, 1::2] *= config.model.image_size / prompt_pil.size[1]
    prompt_label = torch.tensor([args.prompt_label], dtype=torch.long, device=device)
    prompt_type = torch.tensor([0], dtype=torch.long, device=device)

    with torch.no_grad():
        raw = model(prompt_tensor, box.to(device), prompt_label, query_tensor, prompt_type)
        preds = model.predict(
            raw,
            prompt_label,
            image_size=config.model.image_size,
            conf_threshold=config.train.conf_threshold,
            nms_iou_threshold=config.train.nms_iou_threshold,
            max_det=config.train.max_det,
        )[0]

    boxes = preds["boxes"].cpu()
    boxes[:, 0::2] *= query_pil.size[0] / config.model.image_size
    boxes[:, 1::2] *= query_pil.size[1] / config.model.image_size
    preds["boxes"] = boxes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_detection_visualization(query_pil, preds, output_dir / "prediction.png")
    payload = {
        "boxes": preds["boxes"].tolist(),
        "scores": preds["scores"].cpu().tolist(),
        "labels": preds["labels"].cpu().tolist(),
    }
    (output_dir / "prediction.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
