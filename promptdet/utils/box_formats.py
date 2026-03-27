from __future__ import annotations

from typing import Iterable, List


def xyxy_to_yolo_xywh(box: Iterable[float], image_w: float, image_h: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    center_x = x1 + width * 0.5
    center_y = y1 + height * 0.5
    return [
        center_x / max(image_w, 1.0),
        center_y / max(image_h, 1.0),
        width / max(image_w, 1.0),
        height / max(image_h, 1.0),
    ]


def yolo_xywh_to_xyxy(box: Iterable[float], image_w: float, image_h: float) -> List[float]:
    center_x, center_y, width, height = [float(v) for v in box]
    center_x *= image_w
    center_y *= image_h
    width *= image_w
    height *= image_h
    half_w = width * 0.5
    half_h = height * 0.5
    return [
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    ]


def yolo_xywh_to_xyxy_tensor(boxes, image_w: float, image_h: float):
    import torch

    boxes = boxes.clone().float()
    if boxes.numel() == 0:
        return boxes
    center_x = boxes[:, 0] * image_w
    center_y = boxes[:, 1] * image_h
    width = boxes[:, 2] * image_w
    height = boxes[:, 3] * image_h
    half_w = width * 0.5
    half_h = height * 0.5
    boxes[:, 0] = center_x - half_w
    boxes[:, 1] = center_y - half_h
    boxes[:, 2] = center_x + half_w
    boxes[:, 3] = center_y + half_h
    return boxes
