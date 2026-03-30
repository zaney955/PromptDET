from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from promptdet.config import ContextPainterConfig

from .common import ConvBNAct, MLP


class GroundingFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=dim * 4, out_dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ScaleGrounder(nn.Module):
    def __init__(self, channels: int, num_heads: int, depth: int, feature_ensemble_start: int):
        super().__init__()
        self.prompt_proj = nn.Conv2d(channels, channels, 1)
        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.hint_proj = nn.Sequential(
            ConvBNAct(3, max(channels // 2, 32), 3),
            ConvBNAct(max(channels // 2, 32), channels, 3),
        )
        self.prompt_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.query_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.blocks = nn.ModuleList([GroundingFusionBlock(channels, num_heads) for _ in range(depth)])
        self.feature_ensemble_start = feature_ensemble_start
        self.out = ConvBNAct(channels * 2, channels, 3)
        nn.init.normal_(self.prompt_token, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)

    def forward(
        self,
        prompt_feat: torch.Tensor,
        prompt_hint: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        query_feat: torch.Tensor,
        type_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_prompts, channels, height, width = prompt_feat.shape
        prompt_hint = prompt_hint.reshape(batch_size * num_prompts, 3, *prompt_hint.shape[-2:])
        prompt_hint = F.interpolate(prompt_hint, size=(height, width), mode="bilinear", align_corners=False)
        flat_prompt_feat = prompt_feat.reshape(batch_size * num_prompts, channels, height, width)
        prompt_tokens = self.prompt_proj(flat_prompt_feat) + self.hint_proj(prompt_hint)
        prompt_tokens = prompt_tokens.reshape(batch_size, num_prompts, channels, height * width).permute(0, 1, 3, 2)

        query_tokens = self.query_proj(query_feat).unsqueeze(1).expand(batch_size, num_prompts, channels, height, width)
        query_tokens = query_tokens.flatten(3).permute(0, 1, 3, 2)

        prompt_tokens = prompt_tokens + self.prompt_token.view(1, 1, 1, channels) + type_embed.unsqueeze(2)
        query_tokens = query_tokens + self.query_token.view(1, 1, 1, channels) + type_embed.unsqueeze(2)
        pair_mask = prompt_instance_mask.float().view(batch_size, num_prompts, 1, 1)
        tokens = torch.cat([prompt_tokens, query_tokens], dim=2) * pair_mask
        seq_len = height * width
        for layer_idx, block in enumerate(self.blocks):
            flat_tokens = tokens.reshape(batch_size * num_prompts, seq_len * 2, channels)
            flat_tokens = block(flat_tokens)
            tokens = flat_tokens.reshape(batch_size, num_prompts, seq_len * 2, channels) * pair_mask
            if layer_idx >= self.feature_ensemble_start:
                prompt_half, query_half = tokens.split(seq_len, dim=2)
                denom = prompt_instance_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).view(batch_size, 1, 1, 1)
                query_half = (query_half * pair_mask).sum(dim=1, keepdim=True) / denom
                query_half = query_half.expand_as(prompt_half)
                tokens = torch.cat([prompt_half, query_half], dim=2)

        _, query_half = tokens.split(seq_len, dim=2)
        denom = prompt_instance_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).view(batch_size, 1, 1)
        query_half = (query_half * pair_mask).sum(dim=1) / denom
        query_half = query_half.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.out(torch.cat([query_feat, query_half], dim=1))


class BBoxPromptGrounder(nn.Module):
    def __init__(
        self,
        channels: int,
        prompt_dim: int,
        max_prompt_classes: int,
        prompt_types: list[str],
        cfg: ContextPainterConfig,
    ):
        super().__init__()
        self.channels = channels
        self.prompt_dim = prompt_dim
        self.max_prompt_classes = max_prompt_classes
        self.scale = cfg.scale

        self.type_embed = nn.Embedding(len(prompt_types), channels)
        self.scale_grounders = nn.ModuleDict({
            name: ScaleGrounder(channels, cfg.num_heads, cfg.depth, cfg.feature_ensemble_start)
            for name in ("p3", "p4", "p5")
        })
        self.instance_proj = nn.Linear(channels, prompt_dim)
        self.detail_proj = nn.ModuleDict({
            "p3": nn.Linear(channels, prompt_dim),
            "p4": nn.Linear(channels, prompt_dim),
            "p5": nn.Linear(channels, prompt_dim),
        })
        self.slot_head = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, max_prompt_classes + 1, 1),
        )
        self.fg_head = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.quality_head = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )

    def _pool_prompt_features(
        self,
        prompt_feats: Dict[str, torch.Tensor],
        prompt_hint_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        detail = []
        for name in ("p3", "p4", "p5"):
            feat = prompt_feats[name]
            batch_size, num_prompts, channels, height, width = feat.shape
            weight = prompt_hint_maps[:, :, :1].reshape(batch_size * num_prompts, 1, *prompt_hint_maps.shape[-2:])
            weight = F.interpolate(weight, size=(height, width), mode="bilinear", align_corners=False)
            weight = weight.reshape(batch_size, num_prompts, 1, height, width)
            denom = weight.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            pooled_feat = (feat * weight).sum(dim=(-2, -1)) / denom.squeeze(-1).squeeze(-1)
            pooled.append(pooled_feat)
            detail.append(self.detail_proj[name](pooled_feat))
        instance_feat = torch.stack(pooled, dim=0).mean(dim=0)
        instance_global = self.instance_proj(instance_feat)
        detail_tokens = torch.stack(detail, dim=2)
        return instance_global, detail_tokens

    def _aggregate_by_class(
        self,
        instance_global: torch.Tensor,
        instance_detail_tokens: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_prompts, _ = instance_global.shape
        class_indices = prompt_class_indices.clamp(min=0, max=self.max_prompt_classes - 1)
        class_assign = F.one_hot(class_indices, num_classes=self.max_prompt_classes).float()
        class_assign = class_assign * prompt_instance_mask.unsqueeze(-1).float()
        class_counts = class_assign.sum(dim=1).clamp(min=1.0)

        class_prototypes = torch.einsum("bpk,bpd->bkd", class_assign, instance_global) / class_counts.unsqueeze(-1)
        class_detail = torch.einsum("bpk,bptd->bktd", class_assign, instance_detail_tokens)
        class_detail = class_detail / class_counts.unsqueeze(-1).unsqueeze(-1)

        padded_class_mask = torch.zeros(
            (batch_size, self.max_prompt_classes),
            dtype=torch.bool,
            device=prompt_class_mask.device,
        )
        padded_class_mask[:, :prompt_class_mask.shape[1]] = prompt_class_mask
        class_prototypes = class_prototypes * padded_class_mask.unsqueeze(-1)
        class_detail = class_detail * padded_class_mask.unsqueeze(-1).unsqueeze(-1)
        return class_prototypes, class_detail, padded_class_mask

    def _build_priors(
        self,
        fused_feats: Dict[str, torch.Tensor],
        class_mask: torch.Tensor,
        image_size: int,
    ) -> Dict[str, torch.Tensor]:
        slot_logits_low = self.slot_head(fused_feats[self.scale])
        fg_logits_low = self.fg_head(fused_feats[self.scale])
        quality_logits_low = self.quality_head(fused_feats[self.scale])

        slot_logits = F.interpolate(slot_logits_low, size=(image_size, image_size), mode="bilinear", align_corners=False)
        fg_logits = F.interpolate(fg_logits_low, size=(image_size, image_size), mode="bilinear", align_corners=False)
        quality_logits = F.interpolate(
            quality_logits_low,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        valid_slot_mask = torch.cat(
            [
                torch.ones((class_mask.shape[0], 1), dtype=torch.bool, device=class_mask.device),
                class_mask,
            ],
            dim=1,
        )
        slot_logits = slot_logits.masked_fill(~valid_slot_mask.unsqueeze(-1).unsqueeze(-1), -1e4)
        slot_prob = slot_logits.softmax(dim=1)
        fg_prob = (1.0 - slot_prob[:, :1]) * torch.sigmoid(quality_logits)

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
        prompt_hint_maps: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
        query_feats: Dict[str, torch.Tensor],
        prompt_type: torch.Tensor,
        image_size: int,
    ) -> Dict[str, torch.Tensor]:
        type_embed = self.type_embed(prompt_type).unsqueeze(1)
        fused_feats = {
            name: self.scale_grounders[name](
                prompt_feats[name],
                prompt_hint_maps,
                prompt_instance_mask,
                query_feats[name],
                type_embed,
            )
            for name in ("p3", "p4", "p5")
        }

        instance_global, instance_detail = self._pool_prompt_features(prompt_feats, prompt_hint_maps)
        class_prototypes, class_detail_tokens, class_mask = self._aggregate_by_class(
            instance_global,
            instance_detail,
            prompt_class_indices,
            prompt_instance_mask,
            prompt_class_mask,
        )
        priors = self._build_priors(fused_feats, class_mask, image_size=image_size)
        return {
            "fused_feats": fused_feats,
            "class_prototypes": class_prototypes,
            "class_detail_tokens": class_detail_tokens,
            "class_mask": class_mask,
            **priors,
        }
