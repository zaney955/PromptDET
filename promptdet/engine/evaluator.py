from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from promptdet.utils.metrics import aggregate_metrics, compute_map_metrics, match_detections
from promptdet.utils.misc import reduce_tensor, unwrap_model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float,
    pre_score_topk: int,
    local_peak_kernel: int,
    oversize_box_threshold: float,
    oversize_box_gamma: float,
    max_detections: int,
) -> Dict[str, float]:
    model.eval()
    model_without_ddp = unwrap_model(model)
    if hasattr(model_without_ddp, "set_context_prior_strength"):
        model_without_ddp.set_context_prior_strength(1.0)
    items = []
    ap_records = []
    for batch in dataloader:
        prompt_images = batch["prompt_images"].to(device, non_blocking=True)
        prompt_image_mask = batch["prompt_image_mask"].to(device, non_blocking=True)
        prompt_boxes = batch["prompt_boxes"].to(device, non_blocking=True)
        prompt_hint_maps = batch["prompt_hint_maps"].to(device, non_blocking=True)
        prompt_target_maps = batch["prompt_target_maps"].to(device, non_blocking=True)
        prompt_crops = batch["prompt_crops"].to(device, non_blocking=True)
        prompt_class_indices = batch["prompt_class_indices"].to(device, non_blocking=True)
        prompt_source_indices = batch["prompt_source_indices"].to(device, non_blocking=True)
        prompt_instance_mask = batch["prompt_instance_mask"].to(device, non_blocking=True)
        prompt_class_ids = batch["prompt_class_ids"].to(device, non_blocking=True)
        prompt_class_mask = batch["prompt_class_mask"].to(device, non_blocking=True)
        query_image = batch["query_image"].to(device, non_blocking=True)

        raw = model(
            prompt_images,
            prompt_image_mask,
            prompt_boxes,
            prompt_hint_maps,
            prompt_target_maps,
            prompt_crops,
            prompt_class_indices,
            prompt_source_indices,
            prompt_instance_mask,
            prompt_class_mask,
            query_image,
        )
        preds = model_without_ddp.predict(
            raw,
            prompt_class_ids,
            prompt_class_mask,
            image_size=query_image.shape[-1],
            score_threshold=score_threshold,
            pre_score_topk=pre_score_topk,
            local_peak_kernel=local_peak_kernel,
            oversize_box_threshold=oversize_box_threshold,
            oversize_box_gamma=oversize_box_gamma,
            max_detections=max_detections,
        )
        for pred, target in zip(preds, batch["targets"]):
            items.append(
                match_detections(
                    pred["boxes"].cpu(),
                    pred["scores"].cpu(),
                    pred["labels"].cpu(),
                    target["boxes"].cpu(),
                    target["category_ids"].cpu(),
                    target["non_target_boxes"].cpu(),
                )
            )
            ap_records.append({
                "pred_boxes": pred["boxes"].cpu(),
                "pred_scores": pred["scores"].cpu(),
                "pred_labels": pred["labels"].cpu(),
                "gt_boxes": target["boxes"].cpu(),
                "gt_labels": target["category_ids"].cpu(),
            })

    metrics = aggregate_metrics(items)
    map_metrics = compute_map_metrics(ap_records)
    count_tensor = torch.tensor(
        [
            metrics.tp,
            metrics.fp,
            metrics.fn,
            metrics.fp_on_non_target_object,
            metrics.fp_near_prompt_target,
            metrics.fp_background,
        ],
        dtype=torch.float32,
        device=device,
    )
    map_tensor = torch.tensor(
        [
            map_metrics["map_10_95"],
            map_metrics["map_50_95"],
            map_metrics["ap50"],
            map_metrics["ap75"],
        ],
        dtype=torch.float32,
        device=device,
    )
    count_tensor = reduce_tensor(count_tensor, average=False)
    map_tensor = reduce_tensor(map_tensor, average=True)
    tp, fp, fn, fp_on_non_target_object, fp_near_prompt_target, fp_background = [float(x) for x in count_tensor.tolist()]
    map_10_95, map_50_95, ap50, ap75 = [float(x) for x in map_tensor.tolist()]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positives_on_non_target_object": fp_on_non_target_object,
        "false_positives_near_prompt_target": fp_near_prompt_target,
        "false_positives_background": fp_background,
        "map_10_95": map_10_95,
        "map_50_95": map_50_95,
        "ap50": ap50,
        "ap75": ap75,
    }
