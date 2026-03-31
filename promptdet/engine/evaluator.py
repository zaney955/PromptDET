from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from promptdet.utils.metrics import aggregate_metrics, match_detections
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
    for batch in dataloader:
        prompt_images = batch["prompt_images"].to(device)
        prompt_boxes = batch["prompt_boxes"].to(device)
        prompt_hint_maps = batch["prompt_hint_maps"].to(device)
        prompt_target_maps = batch["prompt_target_maps"].to(device)
        prompt_class_indices = batch["prompt_class_indices"].to(device)
        prompt_instance_mask = batch["prompt_instance_mask"].to(device)
        prompt_class_ids = batch["prompt_class_ids"].to(device)
        prompt_class_mask = batch["prompt_class_mask"].to(device)
        query_image = batch["query_image"].to(device)

        raw = model(
            prompt_images,
            prompt_boxes,
            prompt_hint_maps,
            prompt_target_maps,
            prompt_class_indices,
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
                )
            )

    metrics = aggregate_metrics(items)
    metric_tensor = torch.tensor(
        [metrics.tp, metrics.fp, metrics.fn],
        dtype=torch.float32,
        device=device,
    )
    metric_tensor = reduce_tensor(metric_tensor, average=False)
    tp, fp, fn = [float(x) for x in metric_tensor.tolist()]
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
    }
