from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from .common import ConvBNAct, MLP


def crop_and_resize(images: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    crops: List[torch.Tensor] = []
    _, _, height, width = images.shape
    for image, box in zip(images, boxes):
        x1, y1, x2, y2 = box.round().long().tolist()
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops, dim=0)


class PromptEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        prompt_dim: int,
        out_channels: int,
        crop_size: int,
        prompt_types: list[str],
        label_dropout: float,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.crop_size = crop_size
        self.label_dropout = label_dropout
        self.prompt_types = {name: idx for idx, name in enumerate(prompt_types)}

        self.crop_encoder = nn.Sequential(
            ConvBNAct(3, 48, 3, stride=2),
            ConvBNAct(48, 96, 3, stride=2),
            ConvBNAct(96, 192, 3, stride=2),
            ConvBNAct(192, out_channels, 3, stride=2),
        )
        self.local_proj = nn.Conv2d(out_channels, out_channels, 1)
        self.global_proj = nn.Linear(out_channels * 2 + 4, prompt_dim)
        self.label_embed = nn.Embedding(num_classes + 1, prompt_dim)
        self.type_embed = nn.Embedding(len(prompt_types), prompt_dim)
        self.scale_proj = nn.ModuleDict({
            "p3": nn.Linear(prompt_dim, out_channels),
            "p4": nn.Linear(prompt_dim, out_channels),
            "p5": nn.Linear(prompt_dim, out_channels),
        })
        self.refine = MLP(prompt_dim, hidden_dim=prompt_dim * 2, out_dim=prompt_dim)

    def forward(
        self,
        support_image: torch.Tensor,
        support_box: torch.Tensor,
        support_feats: Dict[str, torch.Tensor],
        prompt_label: torch.Tensor,
        prompt_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        crop = crop_and_resize(support_image, support_box, self.crop_size)
        crop_feat = self.crop_encoder(crop)
        local_tokens = self.local_proj(crop_feat).flatten(2).transpose(1, 2)
        crop_global = crop_feat.mean(dim=(2, 3))
        support_global = support_feats["p5"].mean(dim=(2, 3))

        box_norm = support_box.clone()
        box_norm[:, 0::2] /= support_image.shape[-1]
        box_norm[:, 1::2] /= support_image.shape[-2]

        prompt_vec = self.global_proj(torch.cat([crop_global, support_global, box_norm], dim=1))

        label_embed = self.label_embed(prompt_label + 1)
        if self.training and self.label_dropout > 0:
            keep = (torch.rand(prompt_label.shape[0], device=prompt_label.device) > self.label_dropout).float().unsqueeze(1)
            label_embed = label_embed * keep
        type_embed = self.type_embed(prompt_type)
        global_prompt = self.refine(prompt_vec + label_embed + type_embed)

        scale_tokens = {name: proj(global_prompt) for name, proj in self.scale_proj.items()}
        return {
            "global": global_prompt,
            "local_tokens": local_tokens,
            "scale_tokens": scale_tokens,
        }
