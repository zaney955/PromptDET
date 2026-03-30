from __future__ import annotations

import random

import torch


def sample_context_colors(
    num_colors: int,
    min_distance: float,
    *,
    rng: random.Random | None = None,
) -> torch.Tensor:
    rng = rng or random
    colors: list[list[float]] = []
    attempts = 0
    while len(colors) < num_colors and attempts < 2048:
        attempts += 1
        candidate = [
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
        ]
        if all(sum((candidate[idx] - color[idx]) ** 2 for idx in range(3)) ** 0.5 >= min_distance for color in colors):
            colors.append(candidate)
    while len(colors) < num_colors:
        hue = len(colors) / max(num_colors, 1)
        colors.append([
            0.2 + 0.8 * hue,
            0.2 + 0.8 * ((hue + 0.37) % 1.0),
            0.2 + 0.8 * ((hue + 0.71) % 1.0),
        ])
    return torch.tensor(colors, dtype=torch.float32)


def _soft_box_region(
    box: torch.Tensor,
    image_size: int,
    sigma_scale: float,
) -> tuple[slice, slice, torch.Tensor]:
    x1, y1, x2, y2 = box.tolist()
    left = max(0, min(int(round(x1)), image_size - 1))
    top = max(0, min(int(round(y1)), image_size - 1))
    right = max(left + 1, min(int(round(x2)), image_size))
    bottom = max(top + 1, min(int(round(y2)), image_size))

    xs = torch.arange(left, right, dtype=torch.float32) + 0.5
    ys = torch.arange(top, bottom, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    half_w = max((right - left) * 0.5, 1.0)
    half_h = max((bottom - top) * 0.5, 1.0)
    sigma_x = max(half_w * sigma_scale, 1.0)
    sigma_y = max(half_h * sigma_scale, 1.0)
    alpha = torch.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    return slice(top, bottom), slice(left, right), alpha


def render_canvas_from_boxes(
    image_size: int,
    boxes: torch.Tensor,
    slot_indices: torch.Tensor,
    colors: torch.Tensor,
    sigma_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canvas = torch.zeros((3, image_size, image_size), dtype=torch.float32)
    label = torch.zeros((image_size, image_size), dtype=torch.long)
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    if boxes.numel() == 0:
        return canvas, label, weight

    for box, slot_idx in zip(boxes, slot_indices):
        top_slice, left_slice, alpha = _soft_box_region(box, image_size, sigma_scale)
        region_weight = weight[top_slice, left_slice]
        region_update = alpha > region_weight
        if not region_update.any():
            continue
        color = colors[int(slot_idx.item())].view(3, 1, 1)
        region_update_3d = region_update.unsqueeze(0).expand(3, -1, -1)
        canvas_region = canvas[:, top_slice, left_slice]
        weighted_color = color * alpha.unsqueeze(0)
        canvas_region[region_update_3d] = weighted_color[region_update_3d]
        canvas[:, top_slice, left_slice] = canvas_region
        label_region = label[top_slice, left_slice]
        label_region[region_update] = int(slot_idx.item()) + 1
        label[top_slice, left_slice] = label_region
        region_weight[region_update] = alpha[region_update]
        weight[top_slice, left_slice] = region_weight
    return canvas, label, weight
