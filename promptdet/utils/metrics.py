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
    fp_on_non_target_object: int = 0
    fp_near_prompt_target: int = 0
    fp_background: int = 0


def match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    non_target_boxes: torch.Tensor | None = None,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    if non_target_boxes is None:
        non_target_boxes = gt_boxes.new_zeros((0, 4))
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return DetectionMetrics(1.0, 1.0, 1.0, 0, 0, 0)
    if pred_boxes.numel() == 0:
        return DetectionMetrics(0.0, 0.0, 0.0, 0, 0, int(gt_boxes.shape[0]))
    if gt_boxes.numel() == 0:
        fp = int(pred_boxes.shape[0])
        fp_on_non_target_object = fp if non_target_boxes.numel() > 0 else 0
        fp_background = fp - fp_on_non_target_object
        return DetectionMetrics(
            0.0,
            0.0,
            0.0,
            0,
            fp,
            0,
            fp_on_non_target_object=fp_on_non_target_object,
            fp_near_prompt_target=0,
            fp_background=fp_background,
        )

    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    ious = pairwise_iou(pred_boxes, gt_boxes)
    non_target_ious = pairwise_iou(pred_boxes, non_target_boxes) if non_target_boxes.numel() > 0 else None
    tp = 0
    fp = 0
    fp_on_non_target_object = 0
    fp_near_prompt_target = 0
    fp_background = 0
    for pred_idx in range(pred_boxes.shape[0]):
        same_label = gt_labels == pred_labels[pred_idx]
        if not same_label.any():
            fp += 1
            best_target_iou = float(ious[pred_idx].max().item()) if gt_boxes.numel() > 0 else 0.0
            best_non_target_iou = (
                float(non_target_ious[pred_idx].max().item())
                if non_target_ious is not None and non_target_boxes.numel() > 0
                else 0.0
            )
            if best_non_target_iou >= iou_threshold:
                fp_on_non_target_object += 1
            elif best_target_iou >= iou_threshold:
                fp_near_prompt_target += 1
            else:
                fp_background += 1
            continue
        candidate_ious = ious[pred_idx].clone()
        candidate_ious[~same_label] = -1
        gt_idx = torch.argmax(candidate_ious)
        best_iou = candidate_ious[gt_idx]
        if best_iou >= iou_threshold and not matched_gt[gt_idx]:
            matched_gt[gt_idx] = True
            tp += 1
        else:
            fp += 1
            best_target_iou = float(ious[pred_idx].max().item()) if gt_boxes.numel() > 0 else 0.0
            best_non_target_iou = (
                float(non_target_ious[pred_idx].max().item())
                if non_target_ious is not None and non_target_boxes.numel() > 0
                else 0.0
            )
            if best_non_target_iou >= iou_threshold:
                fp_on_non_target_object += 1
            elif best_target_iou >= iou_threshold:
                fp_near_prompt_target += 1
            else:
                fp_background += 1
    fn = int((~matched_gt).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return DetectionMetrics(
        precision,
        recall,
        f1,
        tp,
        fp,
        fn,
        fp_on_non_target_object=fp_on_non_target_object,
        fp_near_prompt_target=fp_near_prompt_target,
        fp_background=fp_background,
    )


def aggregate_metrics(items: List[DetectionMetrics]) -> DetectionMetrics:
    if not items:
        return DetectionMetrics(0.0, 0.0, 0.0, 0, 0, 0)
    tp = sum(item.tp for item in items)
    fp = sum(item.fp for item in items)
    fn = sum(item.fn for item in items)
    fp_on_non_target_object = sum(item.fp_on_non_target_object for item in items)
    fp_near_prompt_target = sum(item.fp_near_prompt_target for item in items)
    fp_background = sum(item.fp_background for item in items)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return DetectionMetrics(
        precision,
        recall,
        f1,
        tp,
        fp,
        fn,
        fp_on_non_target_object=fp_on_non_target_object,
        fp_near_prompt_target=fp_near_prompt_target,
        fp_background=fp_background,
    )
