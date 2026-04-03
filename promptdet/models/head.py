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
        self.prior_fuse = nn.ModuleDict()
        self.class_prior_fuse = nn.ModuleDict()
        self.box_heads = nn.ModuleDict()
        self.objectness_heads = nn.ModuleDict()
        self.targetness_heads = nn.ModuleDict()
        self.class_heads = nn.ModuleDict()
        for name in self.scales:
            self.prior_fuse[name] = ConvBNAct(channels + 1, channels, 3)
            self.class_prior_fuse[name] = ConvBNAct(channels + 2, channels, 3)
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
            self.targetness_heads[name] = nn.Sequential(
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
        self.one2one_targetness_heads = copy.deepcopy(self.targetness_heads)
        self.one2one_class_heads = copy.deepcopy(self.class_heads)
        self.strides = {"p3": 8, "p4": 16, "p5": 32}

    def _forward_branch(
        self,
        shared_feats: Dict[str, torch.Tensor],
        class_feats: Dict[str, torch.Tensor],
        box_heads: nn.ModuleDict,
        objectness_heads: nn.ModuleDict,
        targetness_heads: nn.ModuleDict,
        class_heads: nn.ModuleDict,
        fg_prior_pyramid: Dict[str, torch.Tensor] | None,
        slot_prior_pyramid: Dict[str, torch.Tensor] | None,
    ) -> Dict[str, List[torch.Tensor]]:
        box_logits = []
        objectness_logits = []
        targetness_logits = []
        class_embeddings = []
        slot_prior_maps = []
        roi_feature_maps = {}
        feature_shapes = []
        strides = []
        for name in self.scales:
            feat = shared_feats[name]
            class_feat = class_feats[name]
            batch_size, num_classes, channels, height, width = class_feat.shape
            fg_prior = feat.new_zeros((batch_size, 1, height, width))
            if fg_prior_pyramid is not None:
                fg_prior = fg_prior_pyramid[name]
                feat = self.prior_fuse[name](torch.cat([feat, fg_prior], dim=1))
            box_logits.append(box_heads[name](feat))

            if slot_prior_pyramid is not None:
                slot_prior_map = slot_prior_pyramid[name]
                class_prior = slot_prior_map[:, 1:]
                slot_prior_maps.append(slot_prior_map)
            else:
                class_prior = feat.new_zeros((batch_size, num_classes, height, width))

            class_fg_prior = fg_prior.unsqueeze(1).expand(-1, num_classes, -1, -1, -1)
            refined_class_feat = self.class_prior_fuse[name](
                torch.cat(
                    [
                        class_feat,
                        class_fg_prior,
                        class_prior.unsqueeze(2),
                    ],
                    dim=2,
                ).reshape(batch_size * num_classes, channels + 2, height, width)
            ).reshape(batch_size, num_classes, channels, height, width)
            flat_class_feat = refined_class_feat.reshape(batch_size * num_classes, channels, height, width)
            objectness_logits.append(
                objectness_heads[name](flat_class_feat).reshape(batch_size, num_classes, 1, height, width)
            )
            targetness_logits.append(
                targetness_heads[name](flat_class_feat).reshape(batch_size, num_classes, 1, height, width)
            )
            class_embeddings.append(
                class_heads[name](flat_class_feat).reshape(batch_size, num_classes, -1, height, width)
            )
            roi_feature_maps[name] = refined_class_feat
            feature_shapes.append(feat.shape[-2:])
            strides.append(self.strides[name])
        return {
            "box_logits": box_logits,
            "objectness_logits": objectness_logits,
            "targetness_logits": targetness_logits,
            "class_embeddings": class_embeddings,
            "slot_prior_maps": slot_prior_maps,
            "roi_feature_maps": roi_feature_maps,
            "feature_shapes": feature_shapes,
            "strides": strides,
        }

    def forward(
        self,
        shared_feats: Dict[str, torch.Tensor],
        class_feats: Dict[str, torch.Tensor],
        fg_prior_pyramid: Dict[str, torch.Tensor] | None = None,
        slot_prior_pyramid: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        return {
            "one2many": self._forward_branch(
                shared_feats,
                class_feats,
                self.box_heads,
                self.objectness_heads,
                self.targetness_heads,
                self.class_heads,
                fg_prior_pyramid,
                slot_prior_pyramid,
            ),
            "one2one": self._forward_branch(
                shared_feats,
                class_feats,
                self.one2one_box_heads,
                self.one2one_objectness_heads,
                self.one2one_targetness_heads,
                self.one2one_class_heads,
                fg_prior_pyramid,
                slot_prior_pyramid,
            ),
        }
