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


def _average_precision(tp: torch.Tensor, fp: torch.Tensor, num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    tp_cum = torch.cumsum(tp.float(), dim=0)
    fp_cum = torch.cumsum(fp.float(), dim=0)
    recall = tp_cum / max(float(num_gt), 1.0)
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)
    mrec = torch.cat([recall.new_tensor([0.0]), recall, recall.new_tensor([1.0])])
    mpre = torch.cat([precision.new_tensor([0.0]), precision, precision.new_tensor([0.0])])
    for idx in range(mpre.numel() - 1, 0, -1):
        mpre[idx - 1] = torch.maximum(mpre[idx - 1], mpre[idx])
    change = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    if change.numel() == 0:
        return 0.0
    return float(((mrec[change + 1] - mrec[change]) * mpre[change + 1]).sum().item())


def compute_map_metrics(
    records: List[dict[str, torch.Tensor]],
    iou_thresholds: torch.Tensor | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.10, 1.00, 0.05)
    if not records:
        return {
            "map_10_95": 0.0,
            "map_50_95": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
        }

    classes = sorted({
        int(label.item())
        for record in records
        for label in record["gt_labels"]
    })
    if not classes:
        return {
            "map_10_95": 0.0,
            "map_50_95": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
        }

    ap_by_threshold: dict[float, list[float]] = {float(thr): [] for thr in iou_thresholds.tolist()}
    for class_id in classes:
        gt_by_image: dict[int, torch.Tensor] = {}
        total_gt = 0
        predictions: list[tuple[float, int, torch.Tensor]] = []
        for image_idx, record in enumerate(records):
            gt_mask = record["gt_labels"] == class_id
            gt_boxes = record["gt_boxes"][gt_mask]
            gt_by_image[image_idx] = gt_boxes
            total_gt += int(gt_boxes.shape[0])

            pred_mask = record["pred_labels"] == class_id
            for score, box in zip(record["pred_scores"][pred_mask], record["pred_boxes"][pred_mask]):
                predictions.append((float(score.item()), image_idx, box))
        if total_gt == 0:
            continue
        predictions.sort(key=lambda item: item[0], reverse=True)
        for threshold in iou_thresholds.tolist():
            matched = {
                image_idx: torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
                for image_idx, gt_boxes in gt_by_image.items()
            }
            tp = []
            fp = []
            for _, image_idx, box in predictions:
                gt_boxes = gt_by_image[image_idx]
                if gt_boxes.numel() == 0:
                    tp.append(0)
                    fp.append(1)
                    continue
                ious = pairwise_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= threshold and not matched[image_idx][best_idx]:
                    matched[image_idx][best_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            if tp:
                ap = _average_precision(torch.tensor(tp), torch.tensor(fp), total_gt)
            else:
                ap = 0.0
            ap_by_threshold[float(threshold)].append(ap)

    def _mean_ap(min_iou: float) -> float:
        values = [
            sum(ap_by_threshold[thr]) / max(len(ap_by_threshold[thr]), 1)
            for thr in ap_by_threshold
            if thr >= min_iou
        ]
        return float(sum(values) / max(len(values), 1))

    def _ap_at(target_iou: float) -> float:
        threshold = min(ap_by_threshold.keys(), key=lambda value: abs(value - target_iou))
        return float(sum(ap_by_threshold[threshold]) / max(len(ap_by_threshold[threshold]), 1))

    return {
        "map_10_95": _mean_ap(0.10),
        "map_50_95": _mean_ap(0.50),
        "ap50": _ap_at(0.50),
        "ap75": _ap_at(0.75),
    }


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
