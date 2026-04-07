from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .common import ConvBNAct, MLP


def crop_and_resize(images: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    crops = []
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


def roi_pool_feature_mean(feats: torch.Tensor, boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    pooled = []
    _, _, feat_h, feat_w = feats.shape
    scale_x = feat_w / max(image_w, 1)
    scale_y = feat_h / max(image_h, 1)
    for feat, box in zip(feats, boxes):
        x1, y1, x2, y2 = box.tolist()
        fx1 = int(max(0, min(round(x1 * scale_x), feat_w - 1)))
        fy1 = int(max(0, min(round(y1 * scale_y), feat_h - 1)))
        fx2 = int(max(fx1 + 1, min(round(x2 * scale_x), feat_w)))
        fy2 = int(max(fy1 + 1, min(round(y2 * scale_y), feat_h)))
        roi = feat[:, fy1:fy2, fx1:fx2].unsqueeze(0)
        pooled.append(F.adaptive_avg_pool2d(roi, output_size=1).flatten(1).squeeze(0))
    return torch.stack(pooled, dim=0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.float()
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=dim, keepdim=False).clamp(min=1.0)
    summed = (values * weights).sum(dim=dim, keepdim=False)
    return summed / denom


class PromptEncoder(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        out_channels: int,
        crop_size: int,
        label_dropout: float,
        max_prompt_classes: int,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.crop_size = crop_size
        self.label_dropout = label_dropout
        self.max_prompt_classes = max_prompt_classes

        self.crop_encoder = nn.Sequential(
            ConvBNAct(3, 48, 3, stride=2),
            ConvBNAct(48, 96, 3, stride=2),
            ConvBNAct(96, 192, 3, stride=2),
            ConvBNAct(192, out_channels, 3, stride=2),
        )
        self.local_proj = nn.Conv2d(out_channels, out_channels, 1)
        self.global_proj = nn.Linear(out_channels * 2 + 4, prompt_dim)
        self.class_slot_embed = nn.Embedding(max_prompt_classes, prompt_dim)
        self.local_slot_proj = nn.Linear(prompt_dim, out_channels)
        self.scale_proj = nn.ModuleDict({
            "p3": nn.Linear(prompt_dim, out_channels),
            "p4": nn.Linear(prompt_dim, out_channels),
            "p5": nn.Linear(prompt_dim, out_channels),
        })
        self.refine = MLP(prompt_dim, hidden_dim=prompt_dim * 2, out_dim=prompt_dim)

    def forward(
        self,
        prompt_images: torch.Tensor,
        prompt_source_indices: torch.Tensor,
        prompt_boxes: torch.Tensor,
        prompt_feats: Dict[str, torch.Tensor],
        prompt_class_indices: torch.Tensor,
        prompt_instance_mask: torch.Tensor,
        prompt_class_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, _, image_h, image_w = prompt_images.shape
        num_instances = prompt_boxes.shape[1]
        if prompt_class_mask.shape[1] > self.max_prompt_classes:
            raise ValueError(
                f"Batch contains {prompt_class_mask.shape[1]} prompt classes, "
                f"but model.max_prompt_classes={self.max_prompt_classes}."
            )

        flat_indices = prompt_source_indices.clamp(min=0)
        gather_index = flat_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            num_instances,
            prompt_images.shape[2],
            image_h,
            image_w,
        )
        instance_images = torch.gather(prompt_images, dim=1, index=gather_index)
        flat_images = instance_images.reshape(batch_size * num_instances, *instance_images.shape[2:])
        flat_boxes = prompt_boxes.reshape(batch_size * num_instances, 4)
        crop = crop_and_resize(flat_images, flat_boxes, self.crop_size)
        crop_feat = self.crop_encoder(crop)
        local_tokens = self.local_proj(crop_feat).flatten(2).transpose(1, 2)
        token_count = local_tokens.shape[1]
        local_tokens = local_tokens.view(batch_size, num_instances, token_count, -1)

        crop_global = crop_feat.mean(dim=(2, 3)).view(batch_size, num_instances, -1)
        roi_globals = []
        flat_prompt_boxes = prompt_boxes.reshape(batch_size * num_instances, 4)
        for name in ("p3", "p4", "p5"):
            flat_feat = prompt_feats[name].reshape(batch_size * num_instances, *prompt_feats[name].shape[2:])
            roi_globals.append(roi_pool_feature_mean(flat_feat, flat_prompt_boxes, image_h, image_w))
        support_global = torch.stack(roi_globals, dim=0).mean(dim=0).view(batch_size, num_instances, -1)

        box_norm = prompt_boxes.clone()
        box_norm[:, :, 0::2] /= max(image_w, 1)
        box_norm[:, :, 1::2] /= max(image_h, 1)

        prompt_vec = self.global_proj(torch.cat([crop_global, support_global, box_norm], dim=-1))

        class_indices = prompt_class_indices.clamp(min=0, max=self.max_prompt_classes - 1)
        class_slot_embed = self.class_slot_embed(class_indices)
        if self.training and self.label_dropout > 0:
            keep = (torch.rand_like(prompt_instance_mask.float()) > self.label_dropout).unsqueeze(-1)
            class_slot_embed = class_slot_embed * keep
        instance_global = self.refine(prompt_vec + class_slot_embed)
        local_tokens = local_tokens + self.local_slot_proj(class_slot_embed).unsqueeze(2)

        instance_mask = prompt_instance_mask.float()
        instance_global = instance_global * instance_mask.unsqueeze(-1)
        local_tokens = local_tokens * instance_mask.unsqueeze(-1).unsqueeze(-1)

        class_assign = F.one_hot(class_indices, num_classes=self.max_prompt_classes).float()
        class_assign = class_assign * instance_mask.unsqueeze(-1)

        class_counts = class_assign.sum(dim=1).clamp(min=1.0)
        class_prototypes = torch.einsum("bpk,bpd->bkd", class_assign, instance_global) / class_counts.unsqueeze(-1)
        class_local_tokens = torch.einsum("bpk,bplc->bklc", class_assign, local_tokens)
        class_local_tokens = class_local_tokens / class_counts.unsqueeze(-1).unsqueeze(-1)
        instance_class_indices = torch.where(
            prompt_instance_mask,
            class_indices,
            torch.full_like(class_indices, -1),
        )

        padded_class_mask = torch.zeros(
            (batch_size, self.max_prompt_classes),
            dtype=torch.bool,
            device=prompt_class_mask.device,
        )
        padded_class_mask[:, :prompt_class_mask.shape[1]] = prompt_class_mask
        class_prototypes = class_prototypes * padded_class_mask.unsqueeze(-1)
        class_local_tokens = class_local_tokens * padded_class_mask.unsqueeze(-1).unsqueeze(-1)
        class_memory_mask = padded_class_mask.unsqueeze(-1).expand(batch_size, self.max_prompt_classes, token_count)
        class_memory_tokens = class_local_tokens.reshape(batch_size, self.max_prompt_classes * token_count, -1)
        class_memory_mask = class_memory_mask.reshape(batch_size, self.max_prompt_classes * token_count)
        instance_memory_mask = prompt_instance_mask.unsqueeze(-1).expand(batch_size, num_instances, token_count)
        instance_memory_tokens = local_tokens.reshape(batch_size, num_instances * token_count, -1)
        instance_memory_mask = instance_memory_mask.reshape(batch_size, num_instances * token_count)
        memory_tokens = torch.cat([class_memory_tokens, instance_memory_tokens], dim=1)
        memory_mask = torch.cat([class_memory_mask, instance_memory_mask], dim=1)

        scale_tokens = {
            name: proj(class_prototypes) * padded_class_mask.unsqueeze(-1)
            for name, proj in self.scale_proj.items()
        }
        instance_scale_tokens = {
            name: proj(instance_global) * prompt_instance_mask.unsqueeze(-1).float()
            for name, proj in self.scale_proj.items()
        }

        return {
            "memory_tokens": memory_tokens,
            "memory_mask": memory_mask,
            "class_prototypes": class_prototypes,
            "class_mask": padded_class_mask,
            "scale_tokens": scale_tokens,
            "instance_prototypes": instance_global,
            "instance_class_indices": instance_class_indices,
            "instance_mask": prompt_instance_mask,
            "instance_scale_tokens": instance_scale_tokens,
        }
