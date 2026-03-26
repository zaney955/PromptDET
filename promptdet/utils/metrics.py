from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .box_ops import pairwise_iou


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return DetectionMetrics(1.0, 1.0, 1.0, 0, 0, 0)
    if pred_boxes.numel() == 0:
        return DetectionMetrics(0.0, 0.0, 0.0, 0, 0, int(gt_boxes.shape[0]))
    if gt_boxes.numel() == 0:
        return DetectionMetrics(0.0, 0.0, 0.0, 0, int(pred_boxes.shape[0]), 0)

    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    ious = pairwise_iou(pred_boxes, gt_boxes)
    tp = 0
    fp = 0
    for pred_idx in range(pred_boxes.shape[0]):
        gt_idx = torch.argmax(ious[pred_idx])
        best_iou = ious[pred_idx, gt_idx]
        if best_iou >= iou_threshold and not matched_gt[gt_idx]:
            matched_gt[gt_idx] = True
            tp += 1
        else:
            fp += 1
    fn = int((~matched_gt).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return DetectionMetrics(precision, recall, f1, tp, fp, fn)


def aggregate_metrics(items: List[DetectionMetrics]) -> DetectionMetrics:
    if not items:
        return DetectionMetrics(0.0, 0.0, 0.0, 0, 0, 0)
    tp = sum(item.tp for item in items)
    fp = sum(item.fp for item in items)
    fn = sum(item.fn for item in items)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return DetectionMetrics(precision, recall, f1, tp, fp, fn)
