from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .common import ConvBNAct, MLP


class PromptQueryFusionBlock(nn.Module):
    def __init__(self, channels: int, prompt_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.prompt_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ffn = MLP(channels, hidden_dim=channels * 4, out_dim=channels)
        self.query_to_prompt = nn.Linear(channels, prompt_dim)
        self.class_film = nn.Linear(prompt_dim, channels * 2)
        self.sim_proj = nn.Conv2d(channels, channels, 1)
        self.null_prompt = nn.Parameter(torch.randn(1, prompt_dim))
        self.null_scale_token = nn.Parameter(torch.randn(1, channels))
        self.refine = ConvBNAct(channels + 6, channels, 3)

    @staticmethod
    def _aggregate_instance_scores(
        instance_scores: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_class_indices: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        safe_indices = instance_class_indices.clamp(min=0, max=num_classes - 1)
        class_assign = F.one_hot(safe_indices, num_classes=num_classes).float()
        class_assign = class_assign * instance_mask.unsqueeze(-1).float()
        class_counts = class_assign.sum(dim=1).clamp(min=1.0)
        class_scores = torch.einsum("bni,bik->bnk", instance_scores, class_assign)
        return class_scores / class_counts.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        class_prototypes: torch.Tensor,
        class_mask: torch.Tensor,
        scale_tokens: torch.Tensor,
        instance_prototypes: torch.Tensor,
        instance_class_indices: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_scale_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        query = self.query_norm(query)
        prompt_tokens = self.prompt_norm(prompt_tokens)
        key_padding_mask = ~prompt_mask.bool()
        attn_out, _ = self.attn(query, prompt_tokens, prompt_tokens, key_padding_mask=key_padding_mask)
        query = query + attn_out
        query = query + self.ffn(query)

        query_prompt = F.normalize(self.query_to_prompt(query), dim=-1, eps=1e-6)
        prompt_proto = F.normalize(class_prototypes, dim=-1, eps=1e-6)
        null_prompt = F.normalize(self.null_prompt, dim=-1, eps=1e-6).expand(b, -1, -1)
        class_logits = torch.einsum("bnd,bkd->bnk", query_prompt, prompt_proto)
        null_prompt_logits = torch.einsum("bnd,bkd->bnk", query_prompt, null_prompt)
        class_vs_null = class_logits - null_prompt_logits
        class_vs_null = class_vs_null.masked_fill(~class_mask.unsqueeze(1), -1e4)
        class_presence = class_vs_null.sigmoid() * class_mask.unsqueeze(1).float()

        instance_prototypes = F.normalize(instance_prototypes, dim=-1, eps=1e-6)
        instance_logits = torch.einsum("bnd,bid->bni", query_prompt, instance_prototypes)
        instance_logits = instance_logits.masked_fill(~instance_mask.unsqueeze(1), -1e4)
        instance_presence = (instance_logits - null_prompt_logits).sigmoid() * instance_mask.unsqueeze(1).float()
        class_instance_presence = self._aggregate_instance_scores(
            instance_presence,
            instance_mask,
            instance_class_indices,
            class_prototypes.shape[1],
        )
        class_instance_presence = class_instance_presence * class_mask.unsqueeze(1).float()

        # Class branches remain independent. Instance-level evidence only reinforces
        # the matching branch of its own class before late aggregation.
        combined_class_presence = 0.5 * (class_presence + class_instance_presence)
        active_class_count = class_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        class_gamma, class_beta = self.class_film(class_prototypes).chunk(2, dim=-1)
        class_gamma = torch.tanh(class_gamma)
        class_branch = (
            query.unsqueeze(2) * (1.0 + class_gamma.unsqueeze(1))
            + class_beta.unsqueeze(1)
        ) * combined_class_presence.unsqueeze(-1)
        query = query + 0.5 * (
            class_branch.max(dim=2).values
            + class_branch.sum(dim=2) / active_class_count.unsqueeze(1)
        )
        x = query.transpose(1, 2).reshape(b, c, h, w)

        sim_feat = self.sim_proj(x)
        feat_tokens = F.normalize(sim_feat.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        scale_tokens = F.normalize(scale_tokens, dim=-1, eps=1e-6)
        null_scale_token = F.normalize(self.null_scale_token, dim=-1, eps=1e-6).expand(b, -1, -1)
        sim_logits = torch.einsum("bnd,bkd->bnk", feat_tokens, scale_tokens)
        null_sim_logits = torch.einsum("bnd,bkd->bnk", feat_tokens, null_scale_token)
        sim_vs_null = (sim_logits - null_sim_logits).masked_fill(~class_mask.unsqueeze(1), -1e4)
        sim_presence = sim_vs_null.sigmoid() * class_mask.unsqueeze(1).float()
        instance_scale_tokens = F.normalize(instance_scale_tokens, dim=-1, eps=1e-6)
        instance_sim_logits = torch.einsum("bnd,bid->bni", feat_tokens, instance_scale_tokens)
        instance_sim_logits = instance_sim_logits.masked_fill(~instance_mask.unsqueeze(1), -1e4)
        instance_sim_presence = (instance_sim_logits - null_sim_logits).sigmoid() * instance_mask.unsqueeze(1).float()
        class_instance_sim_presence = self._aggregate_instance_scores(
            instance_sim_presence,
            instance_mask,
            instance_class_indices,
            class_prototypes.shape[1],
        )
        class_instance_sim_presence = class_instance_sim_presence * class_mask.unsqueeze(1).float()
        combined_sim_presence = 0.5 * (sim_presence + class_instance_sim_presence)

        peak_prompt_presence = combined_class_presence.max(dim=-1).values.reshape(b, 1, h, w)
        mean_prompt_presence = (combined_class_presence.sum(dim=-1) / active_class_count).reshape(b, 1, h, w)
        peak_sim_presence = combined_sim_presence.max(dim=-1).values.reshape(b, 1, h, w)
        mean_sim_presence = (combined_sim_presence.sum(dim=-1) / active_class_count).reshape(b, 1, h, w)
        peak_instance_presence = instance_presence.max(dim=-1).values.reshape(b, 1, h, w)
        peak_instance_sim_presence = instance_sim_presence.max(dim=-1).values.reshape(b, 1, h, w)
        return self.refine(
            torch.cat(
                [
                    x,
                    peak_prompt_presence,
                    mean_prompt_presence,
                    peak_sim_presence,
                    mean_sim_presence,
                    peak_instance_presence,
                    peak_instance_sim_presence,
                ],
                dim=1,
            )
        )


class PromptFusionNeck(nn.Module):
    def __init__(self, channels: int, prompt_dim: int, num_heads: int):
        super().__init__()
        self.blocks = nn.ModuleDict({
            "p3": PromptQueryFusionBlock(channels, prompt_dim, num_heads),
            "p4": PromptQueryFusionBlock(channels, prompt_dim, num_heads),
            "p5": PromptQueryFusionBlock(channels, prompt_dim, num_heads),
        })

    def forward(self, feats: Dict[str, torch.Tensor], prompt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for name, feat in feats.items():
            outputs[name] = self.blocks[name](
                feat,
                prompt["memory_tokens"],
                prompt["memory_mask"],
                prompt["class_prototypes"],
                prompt["class_mask"],
                prompt["scale_tokens"][name],
                prompt["instance_prototypes"],
                prompt["instance_class_indices"],
                prompt["instance_mask"],
                prompt["instance_scale_tokens"][name],
            )
        return outputs
