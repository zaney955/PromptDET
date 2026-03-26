from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def clamp_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, height)
    return boxes


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    lt = torch.maximum(box1[..., :2], box2[..., :2])
    rb = torch.minimum(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = bbox_iou(box1, box2, eps=eps)
    center1 = (box1[..., :2] + box1[..., 2:]) * 0.5
    center2 = (box2[..., :2] + box2[..., 2:]) * 0.5
    center_dist = ((center1 - center2) ** 2).sum(dim=-1)

    enclose_lt = torch.minimum(box1[..., :2], box2[..., :2])
    enclose_rb = torch.maximum(box1[..., 2:], box2[..., 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    c2 = (enclose_wh ** 2).sum(dim=-1) + eps

    wh1 = (box1[..., 2:] - box1[..., :2]).clamp(min=eps)
    wh2 = (box2[..., 2:] - box2[..., :2]).clamp(min=eps)
    v = (4 / torch.pi**2) * (torch.atan(wh2[..., 0] / wh2[..., 1]) - torch.atan(wh1[..., 0] / wh1[..., 1])) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (center_dist / c2 + alpha * v)


def make_anchors(
    feature_shapes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    points = []
    stride_tensor = []
    for (h, w), stride in zip(feature_shapes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        anchor = torch.stack((xs + 0.5, ys + 0.5), dim=-1).reshape(-1, 2) * stride
        points.append(anchor)
        stride_tensor.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))
    return torch.cat(points, dim=0), torch.cat(stride_tensor, dim=0)


def dist2bbox(anchor_points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    x1 = anchor_points[:, 0] - distances[:, 0]
    y1 = anchor_points[:, 1] - distances[:, 1]
    x2 = anchor_points[:, 0] + distances[:, 2]
    y2 = anchor_points[:, 1] + distances[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox2dist(anchor_points: torch.Tensor, boxes: torch.Tensor, reg_max: int, stride_tensor: torch.Tensor) -> torch.Tensor:
    left = (anchor_points[:, 0] - boxes[:, 0]) / stride_tensor.squeeze(-1)
    top = (anchor_points[:, 1] - boxes[:, 1]) / stride_tensor.squeeze(-1)
    right = (boxes[:, 2] - anchor_points[:, 0]) / stride_tensor.squeeze(-1)
    bottom = (boxes[:, 3] - anchor_points[:, 1]) / stride_tensor.squeeze(-1)
    dist = torch.stack((left, top, right, bottom), dim=-1)
    return dist.clamp(min=0, max=reg_max - 1 - 1e-3)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[:1]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = bbox_iou(boxes[i], boxes[order[1:]]).reshape(-1)
        order = order[1:][ious <= iou_threshold]
    return torch.cat(keep)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = []
    for label in labels.unique():
        inds = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        if inds.numel() == 0:
            continue
        sub_keep = nms(boxes[inds], scores[inds], iou_threshold)
        keep.append(inds[sub_keep])
    if not keep:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.cat(keep)


def rescale_boxes(boxes: torch.Tensor, from_size: int, to_width: int, to_height: int) -> torch.Tensor:
    boxes = boxes.clone()
    scale_x = to_width / from_size
    scale_y = to_height / from_size
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    return boxes


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    return inter / (area1 + area2 - inter + 1e-7)

