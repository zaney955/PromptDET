from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .common import ConvBNAct


class PromptDetectHead(nn.Module):
    def __init__(self, channels: int, reg_max: int, prompt_dim: int, scales: List[str] | None = None):
        super().__init__()
        self.reg_max = reg_max
        self.scales = scales or ["p3", "p4", "p5"]
        self.box_heads = nn.ModuleDict()
        self.class_heads = nn.ModuleDict()
        for name in self.scales:
            self.box_heads[name] = nn.Sequential(
                ConvBNAct(channels, channels, 3),
                ConvBNAct(channels, channels, 3),
                nn.Conv2d(channels, 4 * reg_max, 1),
            )
            self.class_heads[name] = nn.Sequential(
                ConvBNAct(channels, channels, 3),
                ConvBNAct(channels, channels, 3),
                nn.Conv2d(channels, prompt_dim, 1),
            )
        self.strides = {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        box_logits = []
        class_embeddings = []
        feature_shapes = []
        strides = []
        for name in self.scales:
            feat = feats[name]
            box_logits.append(self.box_heads[name](feat))
            class_embeddings.append(self.class_heads[name](feat))
            feature_shapes.append(feat.shape[-2:])
            strides.append(self.strides[name])
        return {
            "box_logits": box_logits,
            "class_embeddings": class_embeddings,
            "feature_shapes": feature_shapes,
            "strides": strides,
        }
