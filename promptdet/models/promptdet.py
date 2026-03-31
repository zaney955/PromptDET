from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from promptdet.config import DenseGroundingConfig, ModelConfig
from promptdet.utils.box_ops import clamp_boxes, dist2bbox, make_anchors

from .backbone import PromptDetBackbone
from .bbox_grounder import BBoxPromptGrounder
from .fusion import PromptFusionNeck
from .head import PromptDetectHead
from .neck import PromptDetNeck
from .prompt_encoder import PromptEncoder


def oversize_box_penalty(boxes: torch.Tensor, image_size: int, area_threshold: float) -> torch.Tensor:
    boxes = clamp_boxes(boxes, image_size, image_size)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    area_ratio = (widths * heights) / max(float(image_size * image_size), 1.0)
    edge_touch = (
        (boxes[:, 0] <= 1.0).float()
        + (boxes[:, 1] <= 1.0).float()
        + (boxes[:, 2] >= image_size - 1.0).float()
        + (boxes[:, 3] >= image_size - 1.0).float()
    ) / 4.0
    return (area_ratio - area_threshold).clamp(min=0.0) * (1.0 + edge_touch)


class PromptDET(nn.Module):
    def __init__(self, cfg: ModelConfig, context_cfg: DenseGroundingConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.context_cfg = context_cfg or DenseGroundingConfig(enabled=False)
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
        self.prompt_fusion = PromptFusionNeck(cfg.neck_channels, cfg.prompt_dim, cfg.num_attention_heads)
        self.grounder = BBoxPromptGrounder(
            channels=cfg.neck_channels,
            prompt_dim=cfg.prompt_dim,
            max_prompt_classes=cfg.max_prompt_classes,
            prompt_types=cfg.prompt_types,
            image_size=cfg.image_size,
            cfg=self.context_cfg,
        )
        self.head = PromptDetectHead(cfg.neck_channels, cfg.reg_max, cfg.prompt_dim)
        self.register_buffer("proj_bins", torch.arange(cfg.reg_max, dtype=torch.float32), persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale_init))
        self.context_prior_strength = 1.0
        self.slot_prior_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.null_memory = nn.Parameter(torch.randn(self.context_cfg.slot_memory_tokens, cfg.prompt_dim))

    def set_context_prior_strength(self, strength: float) -> None:
        self.context_prior_strength = float(strength)

    def forward(
        self,
        prompt_images: torch.Tensor,
        prompt_boxes: torch.Tensor,
        prompt_hint_maps: torch.Tensor,
        prompt_target_maps: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
        query_image: torch.Tensor,
        prompt_type: torch.Tensor,
        query_target_map: torch.Tensor | None = None,
        decode: bool = False,
    ) -> Dict[str, Dict[str, List[torch.Tensor]]] | Dict[str, Dict[str, torch.Tensor]]:
        batch_size, num_instances = prompt_images.shape[:2]
        flat_prompt_images = prompt_images.reshape(batch_size * num_instances, *prompt_images.shape[2:]).contiguous()
        query_image = query_image.contiguous()
        flat_prompt_feats = self.neck(self.backbone(flat_prompt_images))
        prompt_feats = {
            name: feat.view(batch_size, num_instances, *feat.shape[1:])
            for name, feat in flat_prompt_feats.items()
        }
        query_feats = self.neck(self.backbone(query_image))
        prompt_encoding = self.prompt_encoder(
            prompt_images,
            prompt_boxes,
            prompt_feats,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
            prompt_type,
        )
        query_feats = self.prompt_fusion(query_feats, prompt_encoding)
        grounding = self.grounder(
            prompt_boxes,
            prompt_hint_maps,
            prompt_feats,
            prompt_target_maps,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
            query_feats,
            query_target_map,
            prompt_type,
        )
        class_detail_tokens = prompt_encoding["class_detail_tokens"]
        bsz, num_slots, num_tokens, dim = class_detail_tokens.shape
        pooled_detail = F.adaptive_avg_pool1d(
            class_detail_tokens.permute(0, 1, 3, 2).reshape(bsz * num_slots, dim, num_tokens),
            output_size=self.context_cfg.slot_memory_tokens,
        ).reshape(bsz, num_slots, dim, self.context_cfg.slot_memory_tokens).permute(0, 1, 3, 2)
        slot_memory = torch.cat([grounding["slot_memory"], pooled_detail], dim=2)
        branches = self.head(
            grounding["fused_feats"],
            fg_prior_pyramid=grounding["fg_prior_pyramid"],
            slot_prior_pyramid=grounding["slot_prior_pyramid"],
        )
        max_scale = math.exp(self.cfg.max_logit_scale)
        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=max_scale)
        for branch in branches.values():
            branch["slot_memory"] = slot_memory
            branch["class_mask"] = grounding["class_mask"]
            branch["logit_scale"] = logit_scale
        branches["context_aux"] = {
            "slot_logits": grounding["slot_logits"],
            "fg_logits": grounding["fg_logits"],
            "center_logits": grounding["center_logits"],
            "slot_prior_map": grounding["slot_prior_map"],
            "query_target_pred": grounding["query_target_pred"],
            "query_target_mask": grounding["query_target_mask"],
            "masked_query_target": grounding["masked_query_target"],
        }
        if decode:
            return self.decode_raw(branches)
        return branches

    def _decode_branch(self, outputs: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        box_logits = outputs["box_logits"]
        objectness_logits = outputs["objectness_logits"]
        targetness_logits = outputs["targetness_logits"]
        class_embeddings = outputs["class_embeddings"]
        slot_prior_maps = outputs.get("slot_prior_maps", [])
        slot_memory = outputs["slot_memory"]
        class_mask = outputs["class_mask"]
        logit_scale = outputs["logit_scale"]

        batch = box_logits[0].shape[0]
        device = box_logits[0].device
        feature_shapes = [(h, w) for h, w in outputs["feature_shapes"]]
        anchor_points, _ = make_anchors(feature_shapes, outputs["strides"], device=device)

        box_parts = []
        objectness_parts = []
        targetness_parts = []
        score_parts = []
        stride_parts = []
        prior_parts = []
        flat_box_logits = []
        slot_memory = F.normalize(slot_memory, dim=-1)
        null_memory = F.normalize(self.null_memory, dim=-1)
        null_parts = []
        for level_idx, (box_logit, objectness_logit, targetness_logit, class_embedding, stride) in enumerate(zip(
            box_logits,
            objectness_logits,
            targetness_logits,
            class_embeddings,
            outputs["strides"],
        )):
            b, _, h, w = box_logit.shape
            box_logit = box_logit.view(b, 4, self.cfg.reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(
                b, h * w, 4, self.cfg.reg_max
            )
            probs = box_logit.softmax(dim=-1)
            dist = (probs * self.proj_bins.to(device)).sum(dim=-1) * stride
            box_parts.append(dist)
            flat_box_logits.append(box_logit)
            flat_objectness = objectness_logit.flatten(2).transpose(1, 2).squeeze(-1)
            objectness_parts.append(flat_objectness)
            flat_targetness = targetness_logit.flatten(2).transpose(1, 2).squeeze(-1)
            targetness_parts.append(flat_targetness)
            flat_embeddings = F.normalize(class_embedding.flatten(2).transpose(1, 2), dim=-1)
            level_scores = torch.einsum("bnd,bktd->bnkt", flat_embeddings, slot_memory).max(dim=-1).values * logit_scale
            if slot_prior_maps:
                flat_slot_prior = slot_prior_maps[level_idx].flatten(2).transpose(1, 2)
                cls_prior = flat_slot_prior[:, :, 1:]
                bg_prior = flat_slot_prior[:, :, :1]
                prior_bias = (
                    cls_prior.clamp(min=1e-4).log()
                    - bg_prior.clamp(min=1e-4).log()
                ) * self.slot_prior_logit_scale * self.context_prior_strength
                level_scores = level_scores + prior_bias
                prior_parts.append(cls_prior)
            else:
                prior_parts.append(level_scores.new_ones(level_scores.shape))
            level_null = torch.einsum("bnd,td->bnt", flat_embeddings, null_memory).max(dim=-1).values * logit_scale
            level_scores = level_scores.masked_fill(~class_mask.unsqueeze(1), -1e4)
            score_parts.append(level_scores)
            null_parts.append(level_null)
            stride_parts.append(torch.full((h * w, 1), stride, device=device))

        pred_dist = torch.cat(box_parts, dim=1)
        pred_objectness = torch.cat(objectness_parts, dim=1)
        pred_boxes = []
        for batch_idx in range(batch):
            pred_boxes.append(dist2bbox(anchor_points, pred_dist[batch_idx]))
        pred_boxes = torch.stack(pred_boxes, dim=0)

        return {
            "pred_boxes": pred_boxes,
            "pred_scores": torch.cat(score_parts, dim=1),
            "pred_objectness": pred_objectness,
            "pred_targetness": torch.cat(targetness_parts, dim=1),
            "pred_null_logits": torch.cat(null_parts, dim=1),
            "pred_slot_priors": torch.cat(prior_parts, dim=1),
            "anchor_points": anchor_points,
            "stride_tensor": torch.cat(stride_parts, dim=0),
            "box_distribution": torch.cat(flat_box_logits, dim=1),
            "class_mask": class_mask,
            "feature_shapes": feature_shapes,
        }

    def decode_raw(
        self,
        outputs: Dict[str, Dict[str, List[torch.Tensor]]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "one2many": self._decode_branch(outputs["one2many"]),
            "one2one": self._decode_branch(outputs["one2one"]),
            "context_aux": outputs.get("context_aux"),
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
        oversize_box_threshold: float = 0.85,
        oversize_box_gamma: float = 20.0,
        max_det: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        # Kept in the public API for config/CLI compatibility. The intended inference path remains NMS-free.
        del nms_iou_threshold
        decoded = (
            outputs
            if "one2one" in outputs and "pred_boxes" in outputs["one2one"]
            else self.decode_raw(outputs)
        )
        branch = decoded["one2one"]
        pred_boxes = branch["pred_boxes"]
        pred_logits = branch["pred_scores"]
        pred_scores = pred_logits.sigmoid()
        pred_objectness = branch["pred_objectness"].sigmoid()
        pred_targetness = branch["pred_targetness"].sigmoid()
        pred_null_logits = branch["pred_null_logits"]
        decoded_class_mask = branch["class_mask"]
        feature_shapes = branch["feature_shapes"]

        results = []
        for batch_idx, (boxes, class_scores, class_logits, class_ids, class_mask) in enumerate(
            zip(pred_boxes, pred_scores, pred_logits, prompt_class_ids, prompt_class_mask)
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
            class_logits_max = valid_logits.gather(1, class_index.unsqueeze(1)).squeeze(1)
            prompt_vs_null = (0.5 * (class_logits_max - pred_null_logits[batch_idx])).sigmoid()
            base_scores = (
                class_scores_max
                * pred_objectness[batch_idx]
                * pred_targetness[batch_idx]
            ).clamp(min=0.0).pow(1.0 / 3.0)
            scores = base_scores * (0.25 + 0.75 * prompt_vs_null)
            if one2one_peak_kernel > 1:
                pad = one2one_peak_kernel // 2
                level_keep = []
                start = 0
                for h, w in feature_shapes:
                    count = h * w
                    score_map = scores[start:start + count].view(1, 1, h, w)
                    peak_mask = F.max_pool2d(
                        score_map,
                        kernel_size=one2one_peak_kernel,
                        stride=1,
                        padding=pad,
                    ).eq(score_map)
                    level_keep.append(peak_mask.view(-1))
                    start += count
                keep = torch.cat(level_keep, dim=0)
            else:
                keep = torch.ones_like(scores, dtype=torch.bool)
            boxes = boxes[keep]
            class_index = class_index[keep]
            scores = scores[keep]
            if scores.numel() > pre_nms_topk:
                topk_scores, topk_idx = torch.topk(scores, k=pre_nms_topk)
                boxes = boxes[topk_idx]
                class_index = class_index[topk_idx]
                scores = topk_scores
            if boxes.numel() > 0:
                size_penalty = oversize_box_penalty(boxes, image_size=image_size, area_threshold=oversize_box_threshold)
                scores = scores * torch.exp(-oversize_box_gamma * size_penalty)
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
