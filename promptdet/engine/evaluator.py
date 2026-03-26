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
    conf_threshold: float,
    nms_iou_threshold: float,
    pre_nms_topk: int,
    one2one_topk: int,
    max_det: int,
) -> Dict[str, float]:
    model.eval()
    model_without_ddp = unwrap_model(model)
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
        preds = model_without_ddp.predict(
            raw,
            prompt_class_ids,
            prompt_class_mask,
            image_size=query_image.shape[-1],
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            pre_nms_topk=pre_nms_topk,
            one2one_topk=one2one_topk,
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
