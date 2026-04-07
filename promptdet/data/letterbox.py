from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class LetterBoxParams:
    orig_w: int
    orig_h: int
    target_size: int
    resized_w: int
    resized_h: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def compute_letterbox_params(orig_w: int, orig_h: int, target_size: int) -> LetterBoxParams:
    orig_w = max(int(orig_w), 1)
    orig_h = max(int(orig_h), 1)
    target_size = max(int(target_size), 1)
    scale = min(target_size / orig_w, target_size / orig_h)
    resized_w = max(int(round(orig_w * scale)), 1)
    resized_h = max(int(round(orig_h * scale)), 1)
    pad_w = max(target_size - resized_w, 0)
    pad_h = max(target_size - resized_h, 0)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return LetterBoxParams(
        orig_w=orig_w,
        orig_h=orig_h,
        target_size=target_size,
        resized_w=resized_w,
        resized_h=resized_h,
        scale=float(scale),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def letterbox_image(
    image: np.ndarray,
    target_size: int,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterBoxParams]:
    params = compute_letterbox_params(image.shape[1], image.shape[0], target_size)
    if image.shape[1] != params.resized_w or image.shape[0] != params.resized_h:
        resized = cv2.resize(image, (params.resized_w, params.resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image
    boxed = cv2.copyMakeBorder(
        resized,
        params.pad_top,
        params.pad_bottom,
        params.pad_left,
        params.pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return np.ascontiguousarray(boxed, dtype=np.uint8), params


def letterbox_boxes(boxes: torch.Tensor, params: LetterBoxParams) -> torch.Tensor:
    boxes = boxes.clone().float()
    if boxes.numel() == 0:
        return boxes
    boxes[:, 0::2] = boxes[:, 0::2] * params.scale + params.pad_left
    boxes[:, 1::2] = boxes[:, 1::2] * params.scale + params.pad_top
    return boxes


def unletterbox_boxes(boxes: torch.Tensor, params: LetterBoxParams) -> torch.Tensor:
    boxes = boxes.clone().float()
    if boxes.numel() == 0:
        return boxes
    boxes[:, 0::2] = (boxes[:, 0::2] - params.pad_left) / max(params.scale, 1e-6)
    boxes[:, 1::2] = (boxes[:, 1::2] - params.pad_top) / max(params.scale, 1e-6)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(params.orig_w))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(params.orig_h))
    return boxes
