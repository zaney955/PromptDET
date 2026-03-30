from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from promptdet.config import ContextPainterConfig

from .common import ConvBNAct, MLP


class ContextPainterBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=dim * 4, out_dim=dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ContextPainter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        prompt_types: list[str],
        cfg: ContextPainterConfig,
    ):
        super().__init__()
        self.image_size = image_size
        self.scale = cfg.scale
        self.dim = cfg.dim
        self.feature_ensemble_start = cfg.feature_ensemble_start

        hidden = max(cfg.dim // 2, 32)
        self.image_proj = nn.Conv2d(in_channels, cfg.dim, 1)
        self.canvas_proj = nn.Sequential(
            ConvBNAct(3, hidden, 3),
            ConvBNAct(hidden, cfg.dim, 3),
        )
        self.output_proj = ConvBNAct(cfg.dim, in_channels, 1)
        self.canvas_decoder = nn.Sequential(
            ConvBNAct(cfg.dim, cfg.dim, 3),
            ConvBNAct(cfg.dim, cfg.dim, 3),
            nn.Conv2d(cfg.dim, 3, 1),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.prompt_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.query_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.type_embed = nn.Embedding(len(prompt_types), cfg.dim)

        self.blocks = nn.ModuleList([ContextPainterBlock(cfg.dim, cfg.num_heads) for _ in range(cfg.depth)])
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.prompt_segment, std=0.02)
        nn.init.normal_(self.query_segment, std=0.02)

    def forward(
        self,
        prompt_feat: torch.Tensor,
        prompt_canvas: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        query_feat: torch.Tensor,
        prompt_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_instances, channels, height, width = prompt_feat.shape
        seq_len = height * width

        flat_prompt_feat = prompt_feat.reshape(batch_size * num_instances, channels, height, width)
        flat_prompt_canvas = prompt_canvas.reshape(batch_size * num_instances, 3, *prompt_canvas.shape[-2:])
        flat_prompt_canvas = F.interpolate(
            flat_prompt_canvas,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        prompt_tokens = self.image_proj(flat_prompt_feat) + self.canvas_proj(flat_prompt_canvas)
        prompt_tokens = prompt_tokens.reshape(batch_size, num_instances, self.dim, height, width)

        query_tokens = self.image_proj(query_feat).unsqueeze(1).expand(batch_size, num_instances, self.dim, height, width)
        type_token = self.type_embed(prompt_type).view(batch_size, 1, self.dim, 1, 1)

        prompt_tokens = prompt_tokens + type_token + self.prompt_segment.view(1, 1, self.dim, 1, 1)
        query_tokens = query_tokens + type_token + self.query_segment.view(1, 1, self.dim, 1, 1)
        query_tokens = query_tokens + self.mask_token.view(1, 1, self.dim, 1, 1)

        prompt_tokens = prompt_tokens.flatten(3).permute(0, 1, 3, 2)
        query_tokens = query_tokens.flatten(3).permute(0, 1, 3, 2)
        tokens = torch.cat([prompt_tokens, query_tokens], dim=2)

        pair_mask = prompt_instance_mask.float().view(batch_size, num_instances, 1, 1)
        tokens = tokens * pair_mask
        for layer_idx, block in enumerate(self.blocks):
            tokens = block(tokens.reshape(batch_size * num_instances, 2 * seq_len, self.dim))
            tokens = tokens.reshape(batch_size, num_instances, 2 * seq_len, self.dim) * pair_mask
            if layer_idx >= self.feature_ensemble_start:
                prompt_half, query_half = tokens.split(seq_len, dim=2)
                denom = prompt_instance_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).view(batch_size, 1, 1, 1)
                avg_query = (query_half * pair_mask).sum(dim=1, keepdim=True) / denom
                query_half = avg_query.expand_as(query_half)
                tokens = torch.cat([prompt_half, query_half], dim=2)

        _, query_half = tokens.split(seq_len, dim=2)
        denom = prompt_instance_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).view(batch_size, 1, 1)
        query_half = (query_half * pair_mask).sum(dim=1) / denom
        query_latent = query_half.transpose(1, 2).reshape(batch_size, self.dim, height, width)

        query_context_feat = self.output_proj(query_latent)
        query_canvas_pred = self.canvas_decoder(query_latent)
        query_canvas_pred = F.interpolate(
            query_canvas_pred,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        return {
            "query_context_feat": query_context_feat,
            "query_canvas_pred_rgb": query_canvas_pred,
        }
