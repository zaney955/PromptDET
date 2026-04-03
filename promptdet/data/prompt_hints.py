from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image
import torch

PROMPT_TARGET_CHANNELS = 5


def _clamp_box(box: torch.Tensor, image_size: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    left = max(0, min(int(round(x1)), image_size - 1))
    top = max(0, min(int(round(y1)), image_size - 1))
    right = max(left + 1, min(int(round(x2)), image_size))
    bottom = max(top + 1, min(int(round(y2)), image_size))
    return left, top, right, bottom


def build_prompt_hint_map(
    image_size: int,
    box: torch.Tensor,
    inner_shrink: float,
    bg_expand: float,
) -> torch.Tensor:
    left, top, right, bottom = _clamp_box(box, image_size)
    width = max(right - left, 1)
    height = max(bottom - top, 1)

    shrink_x = max(int(round(width * (1.0 - inner_shrink) * 0.5)), 1)
    shrink_y = max(int(round(height * (1.0 - inner_shrink) * 0.5)), 1)
    inner_left = min(max(left + shrink_x, left), right - 1)
    inner_top = min(max(top + shrink_y, top), bottom - 1)
    inner_right = max(inner_left + 1, min(right - shrink_x, right))
    inner_bottom = max(inner_top + 1, min(bottom - shrink_y, bottom))

    expand_x = max(int(round(width * bg_expand)), 1)
    expand_y = max(int(round(height * bg_expand)), 1)
    outer_left = max(0, left - expand_x)
    outer_top = max(0, top - expand_y)
    outer_right = min(image_size, right + expand_x)
    outer_bottom = min(image_size, bottom + expand_y)

    fg_seed = torch.zeros((image_size, image_size), dtype=torch.float32)
    unknown = torch.zeros((image_size, image_size), dtype=torch.float32)
    bg_seed = torch.ones((image_size, image_size), dtype=torch.float32)

    unknown[top:bottom, left:right] = 1.0
    fg_seed[inner_top:inner_bottom, inner_left:inner_right] = 1.0
    bg_seed[outer_top:outer_bottom, outer_left:outer_right] = 0.0
    bg_seed[fg_seed > 0] = 0.0
    unknown[fg_seed > 0] = 0.0
    return torch.stack([fg_seed, unknown, bg_seed], dim=0)


def _box_center_heatmap(
    image_size: int,
    box: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    x1, y1, x2, y2 = box.float().tolist()
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    half_w = max(0.5 * (x2 - x1), 1.0)
    half_h = max(0.5 * (y2 - y1), 1.0)
    ys, xs = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    dx = (xs - center_x) / (half_w * sigma)
    dy = (ys - center_y) / (half_h * sigma)
    heat = torch.exp(-0.5 * (dx * dx + dy * dy))
    left, top, right, bottom = _clamp_box(box, image_size)
    box_mask = torch.zeros((image_size, image_size), dtype=torch.float32)
    box_mask[top:bottom, left:right] = 1.0
    return heat * box_mask


def build_prompt_target_map(
    image_size: int,
    box: torch.Tensor,
    slot_index: int,
    slot_colors: torch.Tensor,
    center_sigma: float,
) -> torch.Tensor:
    target = torch.zeros((PROMPT_TARGET_CHANNELS, image_size, image_size), dtype=torch.float32)
    left, top, right, bottom = _clamp_box(box, image_size)
    box_mask = torch.zeros((image_size, image_size), dtype=torch.float32)
    box_mask[top:bottom, left:right] = 1.0
    if 0 <= slot_index < slot_colors.shape[0]:
        target[:3, box_mask > 0] = slot_colors[slot_index].view(3, 1)
    target[3] = box_mask
    target[4] = _box_center_heatmap(image_size, box, sigma=center_sigma)
    return target


def build_target_map_from_dense_targets(
    slot_target: torch.Tensor,
    fg_target: torch.Tensor,
    center_target: torch.Tensor,
    slot_colors: torch.Tensor,
) -> torch.Tensor:
    height, width = slot_target.shape[-2:]
    target = torch.zeros((PROMPT_TARGET_CHANNELS, height, width), dtype=torch.float32)
    for slot_idx in range(slot_colors.shape[0]):
        mask = slot_target == (slot_idx + 1)
        if mask.any():
            target[:3, mask] = slot_colors[slot_idx].view(3, 1)
    target[3] = fg_target.float()
    target[4] = center_target.float()
    return target


def build_query_detection_targets(
    image_size: int,
    boxes: torch.Tensor,
    slot_indices: torch.Tensor,
    slot_colors: torch.Tensor,
    center_sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    slot_target = torch.zeros((image_size, image_size), dtype=torch.long)
    fg_target = torch.zeros((image_size, image_size), dtype=torch.float32)
    center_target = torch.zeros((image_size, image_size), dtype=torch.float32)
    valid_mask = torch.ones((image_size, image_size), dtype=torch.bool)

    if boxes.numel() > 0:
        areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        order = torch.argsort(areas)
        for box_idx in order.tolist():
            box = boxes[box_idx]
            slot = int(slot_indices[box_idx].item()) + 1
            left, top, right, bottom = _clamp_box(box, image_size)
            mask = torch.zeros((image_size, image_size), dtype=torch.bool)
            mask[top:bottom, left:right] = True
            slot_target[mask] = slot
            fg_target[mask] = 1.0
            center_target = torch.maximum(center_target, _box_center_heatmap(image_size, box, sigma=center_sigma))

    target_map = build_target_map_from_dense_targets(slot_target, fg_target, center_target, slot_colors)
    return slot_target, fg_target, center_target, valid_mask, target_map


def sample_slot_colors(
    num_slots: int,
    min_distance: float = 0.45,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_slots <= 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    base = torch.tensor(
        [
            [1.0, 0.25, 0.25],
            [0.25, 0.7, 1.0],
            [0.35, 1.0, 0.45],
            [1.0, 0.85, 0.3],
            [0.85, 0.45, 1.0],
            [1.0, 0.55, 0.75],
            [0.1, 1.0, 1.0],
            [1.0, 0.5, 0.1],
        ],
        dtype=torch.float32,
    )
    colors: list[torch.Tensor] = []
    for idx in range(num_slots):
        if idx < base.shape[0]:
            candidate = base[idx]
        else:
            accepted = False
            for _ in range(64):
                candidate = torch.rand(3, generator=generator) * 0.85 + 0.1
                if not colors:
                    accepted = True
                    break
                dist = torch.stack([torch.norm(candidate - item, p=2) for item in colors])
                if float(dist.min().item()) >= min_distance:
                    accepted = True
                    break
            if not accepted:
                angle = 2.0 * math.pi * (idx / max(num_slots, 1))
                candidate = torch.tensor(
                    [
                        0.5 + 0.45 * math.cos(angle),
                        0.5 + 0.45 * math.cos(angle + 2.0 * math.pi / 3.0),
                        0.5 + 0.45 * math.cos(angle + 4.0 * math.pi / 3.0),
                    ],
                    dtype=torch.float32,
                ).clamp(0.05, 0.95)
        colors.append(candidate)
    return torch.stack(colors, dim=0)


def resize_mask(mask: torch.Tensor, size: int, mode: str = "nearest") -> torch.Tensor:
    return torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode=mode,
    ).squeeze(0).squeeze(0)


def tensor_to_image(mask: torch.Tensor) -> Image.Image:
    array = mask.detach().cpu().clamp(0.0, 1.0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="L")


def rgb_target_to_image(target_map: torch.Tensor) -> Image.Image:
    rgb = target_map[:3].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()
    return Image.fromarray(rgb, mode="RGB")
