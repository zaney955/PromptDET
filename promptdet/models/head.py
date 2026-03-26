from __future__ import annotations

import copy
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
        self.objectness_heads = nn.ModuleDict()
        self.class_heads = nn.ModuleDict()
        for name in self.scales:
            self.box_heads[name] = nn.Sequential(
                ConvBNAct(channels, channels, 3),
                ConvBNAct(channels, channels, 3),
                nn.Conv2d(channels, 4 * reg_max, 1),
            )
            self.objectness_heads[name] = nn.Sequential(
                ConvBNAct(channels, channels, 3),
                ConvBNAct(channels, channels, 3),
                nn.Conv2d(channels, 1, 1),
            )
            self.class_heads[name] = nn.Sequential(
                ConvBNAct(channels, channels, 3),
                ConvBNAct(channels, channels, 3),
                nn.Conv2d(channels, prompt_dim, 1),
            )
        self.one2one_box_heads = copy.deepcopy(self.box_heads)
        self.one2one_objectness_heads = copy.deepcopy(self.objectness_heads)
        self.one2one_class_heads = copy.deepcopy(self.class_heads)
        self.strides = {"p3": 8, "p4": 16, "p5": 32}

    def _forward_branch(
        self,
        feats: Dict[str, torch.Tensor],
        box_heads: nn.ModuleDict,
        objectness_heads: nn.ModuleDict,
        class_heads: nn.ModuleDict,
    ) -> Dict[str, List[torch.Tensor]]:
        box_logits = []
        objectness_logits = []
        class_embeddings = []
        feature_shapes = []
        strides = []
        for name in self.scales:
            feat = feats[name]
            box_logits.append(box_heads[name](feat))
            objectness_logits.append(objectness_heads[name](feat))
            class_embeddings.append(class_heads[name](feat))
            feature_shapes.append(feat.shape[-2:])
            strides.append(self.strides[name])
        return {
            "box_logits": box_logits,
            "objectness_logits": objectness_logits,
            "class_embeddings": class_embeddings,
            "feature_shapes": feature_shapes,
            "strides": strides,
        }

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        detached_feats = {name: feat.detach() for name, feat in feats.items()}
        return {
            "one2many": self._forward_branch(feats, self.box_heads, self.objectness_heads, self.class_heads),
            "one2one": self._forward_branch(
                detached_feats,
                self.one2one_box_heads,
                self.one2one_objectness_heads,
                self.one2one_class_heads,
            ),
        }
