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


def save_grounding_visualizations(
    grounding_aux: Dict[str, torch.Tensor],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fg_prob = grounding_aux["fg_logits"][0].sigmoid().detach().cpu().squeeze(0)
    Image.fromarray(fg_prob.clamp(0.0, 1.0).mul(255).byte().numpy(), mode="L").save(output_dir / "grounding_fg.png")

    slot_logits = grounding_aux["slot_logits"][0].detach().cpu()
    slot_pred = slot_logits.argmax(dim=0)
    num_slots = int(slot_logits.shape[0] - 1)
    palette = np.zeros((num_slots + 1, 3), dtype=np.uint8)
    base_colors = np.array(
        [
            [0, 0, 0],
            [255, 80, 80],
            [80, 200, 255],
            [120, 255, 120],
            [255, 220, 80],
            [220, 120, 255],
        ],
        dtype=np.uint8,
    )
    palette[: min(len(palette), len(base_colors))] = base_colors[: min(len(palette), len(base_colors))]
    for idx in range(len(base_colors), len(palette)):
        palette[idx] = np.array([(37 * idx) % 255, (97 * idx) % 255, (181 * idx) % 255], dtype=np.uint8)
    slot_rgb = palette[slot_pred.numpy()]
    Image.fromarray(slot_rgb, mode="RGB").save(output_dir / "grounding_slot.png")
