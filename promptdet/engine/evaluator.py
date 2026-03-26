from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from promptdet.utils.metrics import aggregate_metrics, match_detections


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float,
    nms_iou_threshold: float,
    max_det: int,
) -> Dict[str, float]:
    model.eval()
    items = []
    for batch in dataloader:
        prompt_images = batch["prompt_images"].to(device)
        prompt_boxes = batch["prompt_boxes"].to(device)
        prompt_class_indices = batch["prompt_class_indices"].to(device)
        prompt_instance_mask = batch["prompt_instance_mask"].to(device)
        prompt_class_ids = batch["prompt_class_ids"].to(device)
        prompt_class_mask = batch["prompt_class_mask"].to(device)
        prompt_type = batch["prompt_type"].to(device)
        query_image = batch["query_image"].to(device)

        raw = model(
            prompt_images,
            prompt_boxes,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
            query_image,
            prompt_type,
        )
        preds = model.predict(
            raw,
            prompt_class_ids,
            prompt_class_mask,
            image_size=query_image.shape[-1],
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_det=max_det,
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
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": float(metrics.tp),
        "fp": float(metrics.fp),
        "fn": float(metrics.fn),
    }
