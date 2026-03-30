from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from promptdet.config import DenseGroundingConfig

from .common import ConvBNAct
from .context_painter import PromptContextPainter


class BBoxPromptGrounder(nn.Module):
    def __init__(
        self,
        channels: int,
        prompt_dim: int,
        max_prompt_classes: int,
        prompt_types: list[str],
        image_size: int,
        cfg: DenseGroundingConfig,
    ):
        super().__init__()
        self.channels = channels
        self.prompt_dim = prompt_dim
        self.max_prompt_classes = max_prompt_classes
        self.scale = cfg.scale
        self.slot_memory_tokens = cfg.slot_memory_tokens

        self.painter = PromptContextPainter(
            in_channels=channels,
            image_size=image_size,
            max_prompt_classes=max_prompt_classes,
            prompt_types=prompt_types,
            cfg=cfg,
        )
        self.context_fuse = nn.ModuleDict({
            name: ConvBNAct(channels * 2, channels, 3)
            for name in ("p3", "p4", "p5")
        })
        self.memory_proj = nn.Linear(cfg.dim, prompt_dim)

    def _extract_instance_memory_tokens(
        self,
        prompt_memory_feat: torch.Tensor,
        prompt_pseudo_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_prompts, dim, height, width = prompt_memory_feat.shape
        grid = max(int(math.ceil(math.sqrt(max(self.slot_memory_tokens, 1)))), 1)
        masks = prompt_pseudo_masks.reshape(batch_size * num_prompts, 1, *prompt_pseudo_masks.shape[-2:])
        masks = F.interpolate(masks, size=(height, width), mode="bilinear", align_corners=False)
        feats = prompt_memory_feat.reshape(batch_size * num_prompts, dim, height, width)
        pooled_feat = F.adaptive_avg_pool2d(feats * masks, output_size=(grid, grid))
        pooled_mask = F.adaptive_avg_pool2d(masks, output_size=(grid, grid))
        pooled_feat = pooled_feat / pooled_mask.clamp(min=1e-6)
        pooled_feat = pooled_feat.masked_fill(pooled_mask.expand_as(pooled_feat) < 1e-4, 0.0)
        tokens = pooled_feat.flatten(2).transpose(1, 2)
        tokens = tokens[:, : self.slot_memory_tokens]
        return self.memory_proj(tokens).reshape(batch_size, num_prompts, self.slot_memory_tokens, self.prompt_dim)

    def _aggregate_slot_memory(
        self,
        prompt_memory_feat: torch.Tensor,
        prompt_pseudo_masks: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _, _, _ = prompt_memory_feat.shape
        instance_tokens = self._extract_instance_memory_tokens(prompt_memory_feat, prompt_pseudo_masks)
        class_indices = prompt_class_indices.clamp(min=0, max=self.max_prompt_classes - 1)
        class_assign = F.one_hot(class_indices, num_classes=self.max_prompt_classes).float()
        class_assign = class_assign * prompt_instance_mask.unsqueeze(-1).float()
        class_counts = class_assign.sum(dim=1).clamp(min=1.0)
        slot_memory = torch.einsum("bpk,bptd->bktd", class_assign, instance_tokens)
        slot_memory = slot_memory / class_counts.unsqueeze(-1).unsqueeze(-1)

        padded_class_mask = torch.zeros(
            (batch_size, self.max_prompt_classes),
            dtype=torch.bool,
            device=prompt_class_mask.device,
        )
        padded_class_mask[:, :prompt_class_mask.shape[1]] = prompt_class_mask
        slot_memory = slot_memory * padded_class_mask.unsqueeze(-1).unsqueeze(-1)
        return slot_memory, padded_class_mask

    def _build_priors(
        self,
        slot_logits: torch.Tensor,
        fg_logits: torch.Tensor,
        quality_logits: torch.Tensor,
        class_mask: torch.Tensor,
        fused_feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        valid_slot_mask = torch.cat(
            [
                torch.ones((class_mask.shape[0], 1), dtype=torch.bool, device=class_mask.device),
                class_mask,
            ],
            dim=1,
        )
        slot_logits = slot_logits.masked_fill(~valid_slot_mask.unsqueeze(-1).unsqueeze(-1), -1e4)
        slot_prob = slot_logits.softmax(dim=1)
        fg_prob = (1.0 - slot_prob[:, :1]) * torch.sigmoid(fg_logits) * torch.sigmoid(quality_logits)
        slot_prior_pyramid = {
            name: F.interpolate(slot_prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            for name, feat in fused_feats.items()
        }
        fg_prior_pyramid = {
            name: F.interpolate(fg_prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            for name, feat in fused_feats.items()
        }
        return {
            "slot_logits": slot_logits,
            "fg_logits": fg_logits,
            "quality_logits": quality_logits,
            "slot_prior_map": slot_prob,
            "slot_prior_pyramid": slot_prior_pyramid,
            "fg_prior_pyramid": fg_prior_pyramid,
        }

    def forward(
        self,
        prompt_feats: Dict[str, torch.Tensor],
        prompt_target_canvases: torch.Tensor,
        prompt_pseudo_masks: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
        query_feats: Dict[str, torch.Tensor],
        query_target_canvas: torch.Tensor | None,
        prompt_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        painter_outputs = self.painter(
            prompt_feats[self.scale],
            prompt_target_canvases,
            prompt_instance_mask,
            query_feats[self.scale],
            query_target_canvas,
            prompt_type,
        )
        query_context_feat = painter_outputs["query_context_feat"]
        fused_feats = {}
        for name in ("p3", "p4", "p5"):
            resized_context = F.interpolate(
                query_context_feat,
                size=query_feats[name].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused_feats[name] = self.context_fuse[name](torch.cat([query_feats[name], resized_context], dim=1))

        slot_memory, class_mask = self._aggregate_slot_memory(
            painter_outputs["prompt_memory_feat"],
            prompt_pseudo_masks,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
        )
        priors = self._build_priors(
            painter_outputs["slot_logits"],
            painter_outputs["fg_logits"],
            painter_outputs["quality_logits"],
            class_mask,
            fused_feats,
        )
        return {
            "fused_feats": fused_feats,
            "slot_memory": slot_memory,
            "class_mask": class_mask,
            "query_canvas_pred_rgb": painter_outputs["query_canvas_pred_rgb"],
            "query_canvas_mask": painter_outputs["query_canvas_mask"],
            "masked_query_canvas_rgb": painter_outputs["masked_query_canvas_rgb"],
            **priors,
        }
