from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .common import ConvBNAct, MLP


class PromptQueryFusionBlock(nn.Module):
    def __init__(self, channels: int, prompt_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.prompt_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ffn = MLP(channels, hidden_dim=channels * 4, out_dim=channels)
        self.film = nn.Linear(prompt_dim, channels * 2)
        self.sim_proj = nn.Conv2d(channels, channels, 1)
        self.refine = ConvBNAct(channels + 1, channels, 3)

    def forward(self, x: torch.Tensor, local_tokens: torch.Tensor, global_prompt: torch.Tensor, scale_token: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        query = self.query_norm(query)
        prompt_tokens = local_tokens + scale_token.unsqueeze(1)
        prompt_tokens = self.prompt_norm(prompt_tokens)
        attn_out, _ = self.attn(query, prompt_tokens, prompt_tokens)
        query = query + attn_out
        query = query + self.ffn(query)
        x = query.transpose(1, 2).reshape(b, c, h, w)

        gamma, beta = self.film(global_prompt).chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        sim_feat = self.sim_proj(x)
        prompt_vec = scale_token / (scale_token.norm(dim=1, keepdim=True) + 1e-6)
        feat_norm = sim_feat / (sim_feat.norm(dim=1, keepdim=True) + 1e-6)
        sim = (feat_norm * prompt_vec[:, :, None, None]).sum(dim=1, keepdim=True)
        return self.refine(torch.cat([x, sim], dim=1))


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
                prompt["local_tokens"],
                prompt["global"],
                prompt["scale_tokens"][name],
            )
        return outputs
