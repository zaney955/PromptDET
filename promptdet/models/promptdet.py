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
            label_dropout=cfg.label_dropout,
            max_prompt_classes=cfg.max_prompt_classes,
        )
        self.prompt_fusion = PromptFusionNeck(cfg.neck_channels, cfg.prompt_dim, cfg.num_attention_heads)
        self.grounder = BBoxPromptGrounder(
            channels=cfg.neck_channels,
            prompt_dim=cfg.prompt_dim,
            max_prompt_classes=cfg.max_prompt_classes,
            image_size=cfg.image_size,
            cfg=self.context_cfg,
        )
        self.head = PromptDetectHead(cfg.neck_channels, cfg.reg_max, cfg.prompt_dim)
        self.register_buffer("proj_bins", torch.arange(cfg.reg_max, dtype=torch.float32), persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale_init))
        self.context_prior_strength = 1.0
        self.slot_prior_logit_scale = nn.Parameter(torch.tensor(1.0))

    def set_context_prior_strength(self, strength: float) -> None:
        self.context_prior_strength = float(strength)

    @staticmethod
    def _build_local_peak_mask(
        flat_scores: torch.Tensor,
        feature_shapes: list[tuple[int, int]],
        kernel_size: int,
    ) -> torch.Tensor:
        if kernel_size <= 1:
            return torch.ones_like(flat_scores, dtype=torch.bool)
        pad = kernel_size // 2
        level_keep = []
        start = 0
        for h, w in feature_shapes:
            count = h * w
            score_map = flat_scores[start:start + count].view(1, 1, h, w)
            peak_mask = F.max_pool2d(
                score_map,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            ).eq(score_map)
            level_keep.append(peak_mask.view(-1))
            start += count
        return torch.cat(level_keep, dim=0)

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
        )
        fusion_outputs = self.prompt_fusion(query_feats, prompt_encoding)
        grounding = self.grounder(
            prompt_boxes,
            prompt_hint_maps,
            prompt_feats,
            prompt_target_maps,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
            fusion_outputs["shared_feats"],
            query_target_map,
        )
        branches = self.head(
            grounding["fused_feats"],
            fusion_outputs["class_feats"],
            fg_prior_pyramid=grounding["fg_prior_pyramid"],
            slot_prior_pyramid=grounding["slot_prior_pyramid"],
        )
        max_scale = math.exp(self.cfg.max_logit_scale)
        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=max_scale)
        for branch in branches.values():
            branch["class_prototypes"] = prompt_encoding["class_prototypes"]
            branch["class_scale_tokens"] = prompt_encoding["scale_tokens"]
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
        class_prototypes = outputs["class_prototypes"]
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
        class_prototypes = F.normalize(class_prototypes, dim=-1, eps=1e-6)
        roi_feature_maps = outputs["roi_feature_maps"]
        roi_scale_tokens = outputs["class_scale_tokens"]["p3"]
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
            flat_objectness = objectness_logit.squeeze(2).flatten(2).transpose(1, 2)
            objectness_parts.append(flat_objectness)
            flat_targetness = targetness_logit.squeeze(2).flatten(2).transpose(1, 2)
            targetness_parts.append(flat_targetness)
            flat_embeddings = F.normalize(
                class_embedding.permute(0, 3, 4, 1, 2).reshape(b, h * w, class_embedding.shape[1], -1),
                dim=-1,
                eps=1e-6,
            )
            level_scores = (flat_embeddings * class_prototypes.unsqueeze(1)).sum(dim=-1) * logit_scale
            if slot_prior_maps:
                flat_slot_prior = slot_prior_maps[level_idx].flatten(2).transpose(1, 2)
                cls_prior = flat_slot_prior[:, :, 1:]
                bg_prior = flat_slot_prior[:, :, :1]
                prior_bias = (
                    cls_prior.clamp(min=1e-4).log()
                    - bg_prior.clamp(min=1e-4).log()
                ) * self.slot_prior_logit_scale * self.context_prior_strength
                prior_bias = prior_bias.clamp(min=-6.0, max=6.0)
                level_scores = level_scores + prior_bias
                prior_parts.append(cls_prior)
            else:
                prior_parts.append(level_scores.new_ones(level_scores.shape))
            level_scores = level_scores.masked_fill(~class_mask.unsqueeze(1), -1e4)
            score_parts.append(level_scores)
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
            "pred_slot_priors": torch.cat(prior_parts, dim=1),
            "anchor_points": anchor_points,
            "stride_tensor": torch.cat(stride_parts, dim=0),
            "box_distribution": torch.cat(flat_box_logits, dim=1),
            "class_mask": class_mask,
            "feature_shapes": feature_shapes,
            "roi_feature_maps": roi_feature_maps,
            "roi_scale_tokens": roi_scale_tokens,
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
        score_threshold: float = 0.25,
        pre_score_topk: int = 256,
        local_peak_kernel: int = 3,
        oversize_box_threshold: float = 0.85,
        oversize_box_gamma: float = 20.0,
        max_detections: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
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
            valid_objectness = pred_objectness[batch_idx][:, effective_mask]
            valid_targetness = pred_targetness[batch_idx][:, effective_mask]
            if valid_scores.numel() == 0:
                results.append({
                    "boxes": boxes.new_zeros((0, 4)),
                    "scores": boxes.new_zeros((0,)),
                    "labels": class_ids.new_zeros((0,)),
                })
                continue

            per_class_boxes = []
            per_class_scores = []
            per_class_labels = []
            score_matrix = (
                valid_scores
                * valid_objectness
                * valid_targetness
            ).clamp(min=0.0)
            winning_class = score_matrix.argmax(dim=1)
            for class_offset, label in enumerate(valid_class_ids):
                class_assignment = winning_class == class_offset
                if not class_assignment.any():
                    continue
                scores = score_matrix[:, class_offset].masked_fill(~class_assignment, 0.0)
                keep = class_assignment & self._build_local_peak_mask(
                    scores,
                    feature_shapes,
                    local_peak_kernel,
                )
                class_boxes = boxes[keep]
                class_scores_kept = scores[keep]
                if class_scores_kept.numel() > pre_score_topk:
                    topk_scores, topk_idx = torch.topk(class_scores_kept, k=pre_score_topk)
                    class_boxes = class_boxes[topk_idx]
                    class_scores_kept = topk_scores
                if class_boxes.numel() > 0:
                    size_penalty = oversize_box_penalty(
                        class_boxes,
                        image_size=image_size,
                        area_threshold=oversize_box_threshold,
                    )
                    class_scores_kept = class_scores_kept * torch.exp(-oversize_box_gamma * size_penalty)
                keep = class_scores_kept > score_threshold
                if keep.any():
                    kept_boxes = clamp_boxes(class_boxes[keep], image_size, image_size)
                    kept_scores = class_scores_kept[keep]
                    kept_labels = torch.full(
                        (int(kept_scores.shape[0]),),
                        int(label.item()),
                        dtype=class_ids.dtype,
                        device=class_ids.device,
                    )
                    per_class_boxes.append(kept_boxes)
                    per_class_scores.append(kept_scores)
                    per_class_labels.append(kept_labels)

            if not per_class_scores:
                results.append({
                    "boxes": boxes.new_zeros((0, 4)),
                    "scores": boxes.new_zeros((0,)),
                    "labels": class_ids.new_zeros((0,)),
                })
                continue

            boxes = torch.cat(per_class_boxes, dim=0)
            scores = torch.cat(per_class_scores, dim=0)
            labels = torch.cat(per_class_labels, dim=0)
            order = scores.argsort(descending=True)[:max_detections]
            results.append({
                "boxes": boxes[order],
                "scores": scores[order],
                "labels": labels[order],
            })
        return results
