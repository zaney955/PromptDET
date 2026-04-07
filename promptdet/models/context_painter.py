from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from promptdet.config import DenseGroundingConfig
from promptdet.data.prompt_hints import PROMPT_TARGET_CHANNELS

from .common import ConvBNAct, MLP


class ContextPainterBlock(nn.Module):
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


class PromptContextPainter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        max_prompt_classes: int,
        cfg: DenseGroundingConfig,
    ):
        super().__init__()
        self.image_size = image_size
        self.max_prompt_classes = max_prompt_classes
        self.dim = cfg.dim
        self.feature_ensemble_start = min(max(int(cfg.feature_ensemble_start), 0), max(int(cfg.depth) - 1, 0))
        self.query_mask_ratio = float(cfg.query_mask_ratio)
        self.target_channels = PROMPT_TARGET_CHANNELS

        hidden = max(cfg.dim // 2, 32)
        self.image_proj = nn.Conv2d(in_channels, cfg.dim, 1)
        self.target_proj = nn.Sequential(
            ConvBNAct(self.target_channels, hidden, 3),
            ConvBNAct(hidden, cfg.dim, 3),
        )
        self.context_proj = nn.Sequential(
            ConvBNAct(cfg.dim * 2, in_channels, 3),
            ConvBNAct(in_channels, in_channels, 3),
        )
        self.canvas_decoder = nn.Sequential(
            ConvBNAct(cfg.dim, cfg.recon_decoder_dim, 3),
            ConvBNAct(cfg.recon_decoder_dim, cfg.recon_decoder_dim, 3),
            nn.Conv2d(cfg.recon_decoder_dim, self.target_channels, 1),
        )
        self.slot_head = nn.Sequential(
            ConvBNAct(cfg.dim * 2, cfg.dim, 3),
            nn.Conv2d(cfg.dim, max_prompt_classes + 1, 1),
        )
        self.fg_head = nn.Sequential(
            ConvBNAct(cfg.dim * 2, cfg.dim, 3),
            nn.Conv2d(cfg.dim, 1, 1),
        )
        self.center_head = nn.Sequential(
            ConvBNAct(cfg.dim * 2, cfg.dim, 3),
            nn.Conv2d(cfg.dim, 1, 1),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.image_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.target_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.prompt_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.query_segment = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.use_checkpoint = False

        self.blocks = nn.ModuleList([ContextPainterBlock(cfg.dim, cfg.num_heads) for _ in range(cfg.depth)])
        self._init_parameters()

    def _init_parameters(self) -> None:
        for parameter in (
            self.mask_token,
            self.image_segment,
            self.target_segment,
            self.prompt_segment,
            self.query_segment,
        ):
            nn.init.normal_(parameter, std=0.02)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.use_checkpoint = bool(enabled)

    def _make_query_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        has_target: bool,
    ) -> torch.Tensor:
        if not has_target or not self.training:
            return torch.ones((batch_size, 1, height, width), device=device)
        if self.query_mask_ratio >= 1.0:
            return torch.ones((batch_size, 1, height, width), device=device)
        if self.query_mask_ratio <= 0.0:
            return torch.zeros((batch_size, 1, height, width), device=device)
        mask = (torch.rand((batch_size, 1, height, width), device=device) < self.query_mask_ratio).float()
        flat_mask = mask.flatten(1)
        for batch_idx in range(batch_size):
            if not flat_mask[batch_idx].any():
                flat_mask[batch_idx, torch.randint(flat_mask.shape[1], (1,), device=device)] = 1.0
        return flat_mask.view(batch_size, 1, height, width)

    def _classwise_reduce_and_broadcast(
        self,
        tokens: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_indices = prompt_class_indices.clamp(min=0, max=self.max_prompt_classes - 1)
        class_assign = F.one_hot(safe_indices, num_classes=self.max_prompt_classes).float()
        class_assign = class_assign * prompt_instance_mask.unsqueeze(-1).float()
        class_counts = class_assign.sum(dim=1).clamp(min=1.0)
        class_tokens = torch.einsum("bpk,bpsd->bksd", class_assign, tokens) / class_counts.unsqueeze(-1).unsqueeze(-1)
        broadcast_tokens = torch.einsum("bpk,bksd->bpsd", class_assign, class_tokens)
        return class_tokens, broadcast_tokens

    def forward(
        self,
        prompt_feat: torch.Tensor,
        prompt_target_maps: torch.Tensor,
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        query_feat: torch.Tensor,
        query_target_maps: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_instances, channels, height, width = prompt_feat.shape
        seq_len = height * width
        device = prompt_feat.device

        flat_prompt_feat = prompt_feat.reshape(batch_size * num_instances, channels, height, width)
        flat_prompt_target = prompt_target_maps.reshape(
            batch_size * num_instances,
            self.target_channels,
            *prompt_target_maps.shape[-2:],
        )
        flat_prompt_target = F.interpolate(flat_prompt_target, size=(height, width), mode="bilinear", align_corners=False)

        if query_target_maps is None:
            query_target_low = query_feat.new_zeros((batch_size, self.target_channels, height, width))
            has_query_target = False
        else:
            query_target_low = F.interpolate(
                query_target_maps,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            has_query_target = True

        query_mask = self._make_query_mask(batch_size, height, width, device=device, has_target=has_query_target)
        masked_query_target = query_target_low * (1.0 - query_mask)

        prompt_img_tokens = self.image_proj(flat_prompt_feat).reshape(batch_size, num_instances, self.dim, height, width)
        prompt_tgt_tokens = self.target_proj(flat_prompt_target).reshape(batch_size, num_instances, self.dim, height, width)

        query_img_tokens = self.image_proj(query_feat).unsqueeze(1).expand(batch_size, num_instances, self.dim, height, width)
        query_tgt_tokens = self.target_proj(masked_query_target).unsqueeze(1).expand(batch_size, num_instances, self.dim, height, width)
        query_mask_tokens = query_mask.unsqueeze(1).expand(batch_size, num_instances, 1, height, width)

        prompt_img_tokens = (
            prompt_img_tokens
            + self.image_segment.view(1, 1, self.dim, 1, 1)
            + self.prompt_segment.view(1, 1, self.dim, 1, 1)
        )
        prompt_tgt_tokens = (
            prompt_tgt_tokens
            + self.target_segment.view(1, 1, self.dim, 1, 1)
            + self.prompt_segment.view(1, 1, self.dim, 1, 1)
        )
        query_img_tokens = (
            query_img_tokens
            + self.image_segment.view(1, 1, self.dim, 1, 1)
            + self.query_segment.view(1, 1, self.dim, 1, 1)
        )
        query_tgt_tokens = (
            query_tgt_tokens
            + query_mask_tokens * self.mask_token.view(1, 1, self.dim, 1, 1)
            + self.target_segment.view(1, 1, self.dim, 1, 1)
            + self.query_segment.view(1, 1, self.dim, 1, 1)
        )

        prompt_img_seq = prompt_img_tokens.flatten(3).permute(0, 1, 3, 2)
        query_img_seq = query_img_tokens.flatten(3).permute(0, 1, 3, 2)
        prompt_tgt_seq = prompt_tgt_tokens.flatten(3).permute(0, 1, 3, 2)
        query_tgt_seq = query_tgt_tokens.flatten(3).permute(0, 1, 3, 2)
        tokens = torch.cat([prompt_img_seq, query_img_seq, prompt_tgt_seq, query_tgt_seq], dim=2)

        pair_mask = prompt_instance_mask.float().view(batch_size, num_instances, 1, 1)
        tokens = tokens * pair_mask
        for layer_idx, block in enumerate(self.blocks):
            flat_tokens = tokens.reshape(batch_size * num_instances, 4 * seq_len, self.dim)
            if self.training and self.use_checkpoint and flat_tokens.requires_grad:
                flat_tokens = checkpoint(block, flat_tokens, use_reentrant=False)
            else:
                flat_tokens = block(flat_tokens)
            tokens = flat_tokens
            tokens = tokens.reshape(batch_size, num_instances, 4 * seq_len, self.dim) * pair_mask
            if layer_idx >= self.feature_ensemble_start:
                prompt_img_seq, query_img_seq, prompt_tgt_seq, query_tgt_seq = torch.split(tokens, seq_len, dim=2)
                _, query_img_seq = self._classwise_reduce_and_broadcast(
                    query_img_seq,
                    prompt_class_indices,
                    prompt_instance_mask,
                )
                _, query_tgt_seq = self._classwise_reduce_and_broadcast(
                    query_tgt_seq,
                    prompt_class_indices,
                    prompt_instance_mask,
                )
                tokens = torch.cat([prompt_img_seq, query_img_seq, prompt_tgt_seq, query_tgt_seq], dim=2)

        prompt_img_seq, query_img_seq, prompt_tgt_seq, query_tgt_seq = torch.split(tokens, seq_len, dim=2)
        class_query_img_seq, _ = self._classwise_reduce_and_broadcast(
            query_img_seq,
            prompt_class_indices,
            prompt_instance_mask,
        )
        class_query_tgt_seq, _ = self._classwise_reduce_and_broadcast(
            query_tgt_seq,
            prompt_class_indices,
            prompt_instance_mask,
        )
        class_mask = (
            F.one_hot(prompt_class_indices.clamp(min=0, max=self.max_prompt_classes - 1), num_classes=self.max_prompt_classes).float()
            * prompt_instance_mask.unsqueeze(-1).float()
        ).sum(dim=1).gt(0).float()
        class_denom = class_mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        query_img_seq = (class_query_img_seq * class_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / class_denom
        query_tgt_seq = (class_query_tgt_seq * class_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / class_denom
        query_img_latent = query_img_seq.transpose(1, 2).reshape(batch_size, self.dim, height, width)
        query_tgt_latent = query_tgt_seq.transpose(1, 2).reshape(batch_size, self.dim, height, width)
        query_latent = torch.cat([query_img_latent, query_tgt_latent], dim=1)

        prompt_img_latent = prompt_img_seq.transpose(2, 3).reshape(batch_size, num_instances, self.dim, height, width)
        prompt_tgt_latent = prompt_tgt_seq.transpose(2, 3).reshape(batch_size, num_instances, self.dim, height, width)
        prompt_memory_feat = 0.5 * (prompt_img_latent + prompt_tgt_latent)

        query_context_feat = self.context_proj(query_latent)
        query_target_pred = self.canvas_decoder(query_tgt_latent)
        query_target_pred = F.interpolate(
            query_target_pred,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        query_target_pred[:, :3] = query_target_pred[:, :3].sigmoid()
        query_target_pred[:, 3:] = query_target_pred[:, 3:].sigmoid()

        dense_slot_logits = self.slot_head(query_latent)
        dense_fg_logits = self.fg_head(query_latent)
        dense_center_logits = self.center_head(query_latent)
        dense_slot_logits = F.interpolate(
            dense_slot_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        dense_fg_logits = F.interpolate(
            dense_fg_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        dense_center_logits = F.interpolate(
            dense_center_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masked_query_target = F.interpolate(
            masked_query_target,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        query_target_mask = F.interpolate(query_mask, size=(self.image_size, self.image_size), mode="nearest")

        return {
            "query_context_feat": query_context_feat,
            "prompt_memory_feat": prompt_memory_feat,
            "query_target_pred": query_target_pred,
            "query_target_mask": query_target_mask,
            "masked_query_target": masked_query_target,
            "slot_logits": dense_slot_logits,
            "fg_logits": dense_fg_logits,
            "center_logits": dense_center_logits,
        }


ContextPainter = PromptContextPainter
