from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from promptdet.config import ModelConfig
from promptdet.utils.box_ops import batched_nms, clamp_boxes, dist2bbox, make_anchors

from .backbone import PromptDetBackbone
from .fusion import PromptFusionNeck
from .head import PromptDetectHead
from .neck import PromptDetNeck
from .prompt_encoder import PromptEncoder


class PromptDET(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = PromptDetBackbone(cfg.backbone_widths)
        self.neck = PromptDetNeck(self.backbone.out_channels, cfg.neck_channels)
        self.prompt_encoder = PromptEncoder(
            num_classes=cfg.num_classes,
            prompt_dim=cfg.prompt_dim,
            out_channels=cfg.neck_channels,
            crop_size=cfg.prompt_crop_size,
            prompt_types=cfg.prompt_types,
            label_dropout=cfg.label_dropout,
        )
        self.fusion = PromptFusionNeck(cfg.neck_channels, cfg.prompt_dim, cfg.num_attention_heads)
        self.head = PromptDetectHead(cfg.neck_channels, cfg.reg_max)
        self.register_buffer("proj_bins", torch.arange(cfg.reg_max, dtype=torch.float32), persistent=False)

    def forward(
        self,
        support_image: torch.Tensor,
        support_box: torch.Tensor,
        prompt_label: torch.Tensor,
        query_image: torch.Tensor,
        prompt_type: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        support_feats = self.neck(self.backbone(support_image))
        query_feats = self.neck(self.backbone(query_image))
        prompt = self.prompt_encoder(support_image, support_box, support_feats, prompt_label, prompt_type)
        fused = self.fusion(query_feats, prompt)
        outputs = self.head(fused)
        outputs["prompt_embedding"] = prompt["global"]
        return outputs

    def decode_raw(
        self,
        outputs: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        box_logits = outputs["box_logits"]
        match_logits = outputs["match_logits"]
        batch = box_logits[0].shape[0]
        device = box_logits[0].device
        feature_shapes = [(h, w) for h, w in outputs["feature_shapes"]]
        anchor_points, stride_tensor = make_anchors(feature_shapes, outputs["strides"], device=device)

        box_parts = []
        score_parts = []
        stride_parts = []
        flat_box_logits = []
        for box_logit, match_logit, stride in zip(box_logits, match_logits, outputs["strides"]):
            b, _, h, w = box_logit.shape
            box_logit = box_logit.view(b, 4, self.cfg.reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w, 4, self.cfg.reg_max)
            probs = box_logit.softmax(dim=-1)
            dist = (probs * self.proj_bins.to(device)).sum(dim=-1) * stride
            box_parts.append(dist)
            flat_box_logits.append(box_logit)
            score_parts.append(match_logit.flatten(2).transpose(1, 2))
            stride_parts.append(torch.full((h * w, 1), stride, device=device))

        pred_dist = torch.cat(box_parts, dim=1)
        pred_scores = torch.cat(score_parts, dim=1).squeeze(-1)
        pred_boxes = []
        for batch_idx in range(batch):
            pred_boxes.append(dist2bbox(anchor_points, pred_dist[batch_idx]))
        pred_boxes = torch.stack(pred_boxes, dim=0)
        return {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "anchor_points": anchor_points,
            "stride_tensor": torch.cat(stride_parts, dim=0),
            "box_distribution": torch.cat(flat_box_logits, dim=1),
            "prompt_embedding": outputs["prompt_embedding"],
        }

    @torch.no_grad()
    def predict(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        prompt_label: torch.Tensor,
        image_size: int,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_det: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        decoded = self.decode_raw(outputs)
        pred_boxes = decoded["pred_boxes"]
        pred_scores = decoded["pred_scores"].sigmoid()
        results = []
        for boxes, scores, label in zip(pred_boxes, pred_scores, prompt_label):
            keep = scores > conf_threshold
            boxes = clamp_boxes(boxes[keep], image_size, image_size)
            scores = scores[keep]
            labels = torch.full((scores.shape[0],), int(label.item()), dtype=torch.long, device=scores.device)
            keep_idx = batched_nms(boxes, scores, labels, nms_iou_threshold)
            keep_idx = keep_idx[:max_det]
            results.append({
                "boxes": boxes[keep_idx],
                "scores": scores[keep_idx],
                "labels": labels[keep_idx],
            })
        return results
