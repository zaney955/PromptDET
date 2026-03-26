from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from promptdet.config import ModelConfig
from promptdet.utils.box_ops import clamp_boxes, dist2bbox, make_anchors

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
            prompt_dim=cfg.prompt_dim,
            out_channels=cfg.neck_channels,
            crop_size=cfg.prompt_crop_size,
            prompt_types=cfg.prompt_types,
            label_dropout=cfg.label_dropout,
            max_prompt_classes=cfg.max_prompt_classes,
        )
        self.fusion = PromptFusionNeck(cfg.neck_channels, cfg.prompt_dim, cfg.num_attention_heads)
        self.head = PromptDetectHead(cfg.neck_channels, cfg.reg_max, cfg.prompt_dim)
        self.register_buffer("proj_bins", torch.arange(cfg.reg_max, dtype=torch.float32), persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale_init))

    def forward(
        self,
        prompt_images: torch.Tensor,
        prompt_boxes: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
        query_image: torch.Tensor,
        prompt_type: torch.Tensor,
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        batch_size, num_instances = prompt_images.shape[:2]
        flat_prompt_images = prompt_images.reshape(batch_size * num_instances, *prompt_images.shape[2:]).contiguous(
            memory_format=torch.channels_last
        )
        query_image = query_image.contiguous(memory_format=torch.channels_last)
        flat_prompt_feats = self.neck(self.backbone(flat_prompt_images))
        prompt_feats = {
            name: feat.view(batch_size, num_instances, *feat.shape[1:])
            for name, feat in flat_prompt_feats.items()
        }
        query_feats = self.neck(self.backbone(query_image))
        prompt = self.prompt_encoder(
            prompt_images,
            prompt_boxes,
            prompt_feats,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
            prompt_type,
        )
        fused = self.fusion(query_feats, prompt)
        branches = self.head(fused)
        max_scale = math.exp(self.cfg.max_logit_scale)
        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=max_scale)
        for branch in branches.values():
            branch["class_prototypes"] = prompt["class_prototypes"]
            branch["class_mask"] = prompt["class_mask"]
            branch["logit_scale"] = logit_scale
        return branches

    def _decode_branch(self, outputs: Dict[str, List[torch.Tensor]], peak_kernel: int = 3) -> Dict[str, torch.Tensor]:
        box_logits = outputs["box_logits"]
        objectness_logits = outputs["objectness_logits"]
        class_embeddings = outputs["class_embeddings"]
        class_prototypes = outputs["class_prototypes"]
        class_mask = outputs["class_mask"]
        logit_scale = outputs["logit_scale"]

        batch = box_logits[0].shape[0]
        device = box_logits[0].device
        feature_shapes = [(h, w) for h, w in outputs["feature_shapes"]]
        anchor_points, _ = make_anchors(feature_shapes, outputs["strides"], device=device)

        box_parts = []
        objectness_parts = []
        embedding_parts = []
        peak_parts = []
        stride_parts = []
        flat_box_logits = []
        for box_logit, objectness_logit, class_embedding, stride in zip(
            box_logits,
            objectness_logits,
            class_embeddings,
            outputs["strides"],
        ):
            b, _, h, w = box_logit.shape
            box_logit = box_logit.view(b, 4, self.cfg.reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(
                b, h * w, 4, self.cfg.reg_max
            )
            probs = box_logit.softmax(dim=-1)
            dist = (probs * self.proj_bins.to(device)).sum(dim=-1) * stride
            box_parts.append(dist)
            flat_box_logits.append(box_logit)
            objectness_parts.append(objectness_logit.flatten(2).transpose(1, 2).squeeze(-1))
            embedding_parts.append(class_embedding.flatten(2).transpose(1, 2))
            if peak_kernel > 1:
                pad = peak_kernel // 2
                peak_mask = F.max_pool2d(objectness_logit, kernel_size=peak_kernel, stride=1, padding=pad).eq(
                    objectness_logit
                )
            else:
                peak_mask = torch.ones_like(objectness_logit, dtype=torch.bool)
            peak_parts.append(peak_mask.flatten(2).transpose(1, 2).squeeze(-1))
            stride_parts.append(torch.full((h * w, 1), stride, device=device))

        pred_dist = torch.cat(box_parts, dim=1)
        pred_objectness = torch.cat(objectness_parts, dim=1)
        query_embeddings = torch.cat(embedding_parts, dim=1)
        pred_boxes = []
        for batch_idx in range(batch):
            pred_boxes.append(dist2bbox(anchor_points, pred_dist[batch_idx]))
        pred_boxes = torch.stack(pred_boxes, dim=0)

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        class_prototypes = F.normalize(class_prototypes, dim=-1)
        pred_scores = torch.einsum("bnd,bkd->bnk", query_embeddings, class_prototypes) * logit_scale
        pred_scores = pred_scores.masked_fill(~class_mask.unsqueeze(1), -1e4)

        return {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_objectness": pred_objectness,
            "anchor_points": anchor_points,
            "stride_tensor": torch.cat(stride_parts, dim=0),
            "box_distribution": torch.cat(flat_box_logits, dim=1),
            "class_mask": class_mask,
            "peak_mask": torch.cat(peak_parts, dim=1),
        }

    def decode_raw(
        self,
        outputs: Dict[str, Dict[str, List[torch.Tensor]]],
        peak_kernel: int = 3,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "one2many": self._decode_branch(outputs["one2many"], peak_kernel=peak_kernel),
            "one2one": self._decode_branch(outputs["one2one"], peak_kernel=peak_kernel),
        }

    @torch.no_grad()
    def predict(
        self,
        outputs: Dict[str, Dict[str, List[torch.Tensor]]] | Dict[str, Dict[str, torch.Tensor]],
        prompt_class_ids: torch.Tensor,
        prompt_class_mask: torch.Tensor,
        image_size: int,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        pre_nms_topk: int = 256,
        one2one_topk: int = 300,
        one2one_peak_kernel: int = 3,
        class_margin_scale: float = 6.0,
        max_det: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        decoded = (
            outputs
            if "one2one" in outputs and "pred_boxes" in outputs["one2one"]
            else self.decode_raw(outputs, peak_kernel=one2one_peak_kernel)
        )
        branch = decoded["one2one"]
        pred_boxes = branch["pred_boxes"]
        pred_logits = branch["pred_scores"]
        pred_scores = pred_logits.sigmoid()
        pred_objectness = branch["pred_objectness"].sigmoid()
        decoded_class_mask = branch["class_mask"]
        peak_masks = branch["peak_mask"]

        results = []
        for batch_idx, (boxes, class_scores, class_logits, class_ids, class_mask, peak_mask) in enumerate(
            zip(pred_boxes, pred_scores, pred_logits, prompt_class_ids, prompt_class_mask, peak_masks)
        ):
            padded_class_ids = torch.full(
                (self.cfg.max_prompt_classes,),
                -1,
                dtype=class_ids.dtype,
                device=class_ids.device,
            )
            padded_mask = torch.zeros((self.cfg.max_prompt_classes,), dtype=torch.bool, device=class_mask.device)
            limit = min(class_ids.shape[0], self.cfg.max_prompt_classes)
            padded_class_ids[:limit] = class_ids[:limit]
            padded_mask[:limit] = class_mask[:limit]
            effective_mask = padded_mask & decoded_class_mask[batch_idx]

            valid_class_ids = padded_class_ids[effective_mask]
            valid_scores = class_scores[:, effective_mask]
            valid_logits = class_logits[:, effective_mask]
            if valid_scores.numel() == 0:
                results.append({
                    "boxes": boxes.new_zeros((0, 4)),
                    "scores": boxes.new_zeros((0,)),
                    "labels": class_ids.new_zeros((0,)),
                })
                continue

            class_scores_max, class_index = valid_scores.max(dim=1)
            if valid_logits.shape[1] > 1:
                top2_logits = torch.topk(valid_logits, k=2, dim=1).values
                margin_gate = torch.sigmoid((top2_logits[:, 0] - top2_logits[:, 1]) * class_margin_scale)
            else:
                margin_gate = torch.ones_like(class_scores_max)
            class_confidence = class_scores_max * margin_gate
            scores = torch.sqrt((class_confidence * pred_objectness[batch_idx]).clamp(min=0.0))
            keep = peak_mask
            boxes = boxes[keep]
            class_index = class_index[keep]
            scores = scores[keep]
            if scores.numel() > pre_nms_topk:
                topk_scores, topk_idx = torch.topk(scores, k=pre_nms_topk)
                boxes = boxes[topk_idx]
                class_index = class_index[topk_idx]
                scores = topk_scores
            keep = scores > conf_threshold
            boxes = clamp_boxes(boxes[keep], image_size, image_size)
            scores = scores[keep]
            labels = valid_class_ids[class_index[keep]]
            if scores.numel() > one2one_topk:
                scores, top_idx = torch.topk(scores, k=one2one_topk)
                boxes = boxes[top_idx]
                labels = labels[top_idx]
            order = scores.argsort(descending=True)[:max_det]
            results.append({
                "boxes": boxes[order],
                "scores": scores[order],
                "labels": labels[order],
            })
        return results
