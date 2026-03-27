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
        self.film = nn.Linear(prompt_dim, channels * 2)
        self.sim_proj = nn.Conv2d(channels, channels, 1)
        self.refine = ConvBNAct(channels + 2, channels, 3)

    def forward(
        self,
        x: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        class_prototypes: torch.Tensor,
        class_mask: torch.Tensor,
        scale_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        query = self.query_norm(query)
        prompt_tokens = self.prompt_norm(prompt_tokens)
        key_padding_mask = ~prompt_mask.bool()
        attn_out, _ = self.attn(query, prompt_tokens, prompt_tokens, key_padding_mask=key_padding_mask)
        query = query + attn_out
        query = query + self.ffn(query)

        query_prompt = F.normalize(self.query_to_prompt(query), dim=-1)
        prompt_proto = F.normalize(class_prototypes, dim=-1)
        class_logits = torch.einsum("bnd,bkd->bnk", query_prompt, prompt_proto)
        class_logits = class_logits.masked_fill(~class_mask.unsqueeze(1), -1e4)
        class_weights = torch.softmax(class_logits, dim=-1)
        class_weights = class_weights * class_mask.unsqueeze(1).float()
        class_weights = class_weights / class_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adaptive_prompt = torch.einsum("bnk,bkd->bnd", class_weights, class_prototypes)

        gamma, beta = self.film(adaptive_prompt).chunk(2, dim=-1)
        query = query * (1 + gamma) + beta
        x = query.transpose(1, 2).reshape(b, c, h, w)

        sim_feat = self.sim_proj(x)
        feat_tokens = F.normalize(sim_feat.flatten(2).transpose(1, 2), dim=-1)
        scale_tokens = F.normalize(scale_tokens, dim=-1)
        sim_logits = torch.einsum("bnd,bkd->bnk", feat_tokens, scale_tokens)
        sim_logits = sim_logits.masked_fill(~class_mask.unsqueeze(1), -1e4)
        max_sim = sim_logits.max(dim=-1).values
        if scale_tokens.shape[1] > 1:
            top2 = torch.topk(sim_logits, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            multi_class = (class_mask.sum(dim=1, keepdim=True) > 1).expand(-1, max_sim.shape[1])
            margin = torch.where(multi_class, margin, torch.ones_like(margin))
        else:
            margin = torch.ones_like(max_sim)
        max_sim = max_sim.reshape(b, 1, h, w)
        margin = margin.reshape(b, 1, h, w)
        return self.refine(torch.cat([x, max_sim, margin], dim=1))


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
            )
        return outputs
