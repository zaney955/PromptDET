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
        support_image = batch["support_image"].to(device)
        support_box = batch["support_box"].to(device)
        prompt_label = batch["prompt_label"].to(device)
        prompt_type = batch["prompt_type"].to(device)
        query_image = batch["query_image"].to(device)

        raw = model(support_image, support_box, prompt_label, query_image, prompt_type)
        preds = model.predict(raw, prompt_label, image_size=query_image.shape[-1], conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold, max_det=max_det)
        for pred, target in zip(preds, batch["targets"]):
            items.append(match_detections(pred["boxes"].cpu(), pred["scores"].cpu(), target["boxes"].cpu()))

    metrics = aggregate_metrics(items)
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": float(metrics.tp),
        "fp": float(metrics.fp),
        "fn": float(metrics.fn),
    }
