from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from PIL import Image, ImageDraw
import torch


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    label_names: Dict[int, str] | None = None,
    color: tuple[int, int, int] = (255, 80, 80),
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if boxes.numel() == 0:
        return canvas
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        caption = []
        if labels is not None:
            label = int(labels[idx].item())
            caption.append(label_names.get(label, str(label)) if label_names else str(label))
        if scores is not None:
            caption.append(f"{float(scores[idx].item()):.2f}")
        if caption:
            draw.text((x1 + 2, max(0, y1 - 12)), " ".join(caption), fill=color)
    return canvas


def save_detection_visualization(
    image: Image.Image,
    prediction: Dict[str, torch.Tensor],
    output_path: str | Path,
    label_names: Dict[int, str] | None = None,
) -> None:
    rendered = draw_boxes(image, prediction["boxes"], prediction.get("labels"), prediction.get("scores"), label_names=label_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
