from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
import torch

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may be unavailable in some environments.
    cv2 = None


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


def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (image * 255.0).astype(np.uint8)


def generate_grabcut_pseudo_mask(
    image_tensor: torch.Tensor,
    box: torch.Tensor,
    inner_shrink: float,
    bg_expand: float,
    iterations: int,
) -> torch.Tensor | None:
    if cv2 is None:
        return None
    image_size = image_tensor.shape[-1]
    left, top, right, bottom = _clamp_box(box, image_size)
    width = max(right - left, 1)
    height = max(bottom - top, 1)

    mask = np.full((image_size, image_size), cv2.GC_BGD, dtype=np.uint8)
    expand_x = max(int(round(width * bg_expand)), 1)
    expand_y = max(int(round(height * bg_expand)), 1)
    outer_left = max(0, left - expand_x)
    outer_top = max(0, top - expand_y)
    outer_right = min(image_size, right + expand_x)
    outer_bottom = min(image_size, bottom + expand_y)
    mask[outer_top:outer_bottom, outer_left:outer_right] = cv2.GC_PR_BGD
    mask[top:bottom, left:right] = cv2.GC_PR_FGD

    shrink_x = max(int(round(width * (1.0 - inner_shrink) * 0.5)), 1)
    shrink_y = max(int(round(height * (1.0 - inner_shrink) * 0.5)), 1)
    inner_left = min(max(left + shrink_x, left), right - 1)
    inner_top = min(max(top + shrink_y, top), bottom - 1)
    inner_right = max(inner_left + 1, min(right - shrink_x, right))
    inner_bottom = max(inner_top + 1, min(bottom - shrink_y, bottom))
    mask[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            _tensor_to_uint8_image(image_tensor),
            mask,
            None,
            bg_model,
            fg_model,
            iterCount=max(int(iterations), 1),
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except Exception:
        return None

    fg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
    fg_tensor = torch.from_numpy(fg.astype(np.float32))
    box_area = max(float(width * height), 1.0)
    fg_area = float(fg_tensor[top:bottom, left:right].sum().item())
    full_area = float(fg_tensor.sum().item())
    if fg_area < box_area * 0.05 or fg_area > box_area * 0.95:
        return None
    if full_area > image_size * image_size * 0.9:
        return None
    return fg_tensor


def build_query_dense_targets(
    image_tensor: torch.Tensor,
    boxes: torch.Tensor,
    slot_indices: torch.Tensor,
    num_slots: int,
    inner_shrink: float,
    bg_expand: float,
    grabcut_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = image_tensor.shape[-1]
    slot_target = torch.zeros((image_size, image_size), dtype=torch.long)
    fg_target = torch.zeros((image_size, image_size), dtype=torch.float32)
    valid_mask = torch.ones((image_size, image_size), dtype=torch.bool)
    pseudo_mask_bank = torch.zeros((max(boxes.shape[0], 1), image_size, image_size), dtype=torch.float32)

    if boxes.numel() == 0:
        return slot_target, fg_target, valid_mask, pseudo_mask_bank[:0]

    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    order = torch.argsort(areas)
    for out_idx, box_idx in enumerate(order.tolist()):
        box = boxes[box_idx]
        slot = int(slot_indices[box_idx].item()) + 1
        hint = build_prompt_hint_map(image_size, box, inner_shrink=inner_shrink, bg_expand=bg_expand)
        pseudo_mask = generate_grabcut_pseudo_mask(
            image_tensor,
            box,
            inner_shrink=inner_shrink,
            bg_expand=bg_expand,
            iterations=grabcut_iters,
        )
        if pseudo_mask is None:
            pseudo_mask = hint[0].clone()
            box_unknown = hint[1] > 0
            valid_mask[box_unknown] = False
        else:
            pseudo_mask = pseudo_mask.float()
            valid_mask[hint[1] > 0] = False
        fg_region = pseudo_mask > 0.5
        fg_target[fg_region] = 1.0
        slot_target[fg_region] = slot
        pseudo_mask_bank[out_idx] = pseudo_mask

    if num_slots >= 0:
        slot_target = slot_target.clamp(min=0, max=num_slots)
    return slot_target, fg_target, valid_mask, pseudo_mask_bank[: boxes.shape[0]]


def resize_mask(mask: torch.Tensor, size: int, mode: str = "nearest") -> torch.Tensor:
    return torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode=mode,
    ).squeeze(0).squeeze(0)


def tensor_to_image(mask: torch.Tensor) -> Image.Image:
    array = mask.detach().cpu().clamp(0.0, 1.0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="L")
