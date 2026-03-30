from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
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


def save_context_prior_visualizations(
    slot_prior_map: torch.Tensor,
    context_colors: torch.Tensor,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = slot_prior_map.detach().cpu()
    colors = context_colors.detach().cpu().clamp(0.0, 1.0)
    fg_prior = (1.0 - prior[0]).clamp(0.0, 1.0).mul(255).byte().numpy()
    Image.fromarray(fg_prior, mode="L").save(output_dir / "fg_prior.png")

    argmax = prior.argmax(dim=0)
    color_canvas = torch.zeros((3, *argmax.shape), dtype=torch.float32)
    for slot_idx in range(1, prior.shape[0]):
        mask = argmax == slot_idx
        if slot_idx - 1 < colors.shape[0]:
            color_canvas[:, mask] = colors[slot_idx - 1].view(3, 1)
    color_canvas = (color_canvas.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    Image.fromarray(color_canvas, mode="RGB").save(output_dir / "slot_prior_argmax.png")
