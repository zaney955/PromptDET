from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from promptdet.config import ContextPainterConfig, LossConfig

from .box_ops import bbox2dist, bbox_ciou, bbox_iou


@dataclass
class AssignmentResult:
    target_scores: torch.Tensor
    target_boxes: torch.Tensor
    fg_mask: torch.Tensor
    matched_gt_indices: torch.Tensor
    matched_labels: torch.Tensor
    duplicate_mask: torch.Tensor


def sigmoid_varifocal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    prob = logits.sigmoid()
    weight = alpha * prob.pow(gamma) * (1 - targets) + targets
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while weights.dim() < loss.dim():
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp(min=eps)


def build_box_region_weights(
    anchor_points: torch.Tensor,
    boxes: torch.Tensor,
    box_weights: torch.Tensor,
    center_sampling_radius: float,
) -> torch.Tensor:
    num_points = anchor_points.shape[0]
    weights = anchor_points.new_zeros((num_points,))
    if boxes.numel() == 0:
        return weights
    if box_weights.numel() == 0:
        box_weights = boxes.new_ones((boxes.shape[0],))

    for box, box_weight in zip(boxes, box_weights):
        inside = (
            (anchor_points[:, 0] >= box[0])
            & (anchor_points[:, 0] <= box[2])
            & (anchor_points[:, 1] >= box[1])
            & (anchor_points[:, 1] <= box[3])
        )
        if not inside.any():
            continue
        gt_center = (box[:2] + box[2:]) * 0.5
        half_size = ((box[2:] - box[:2]) * 0.5).clamp(min=1.0)
        center_delta = (anchor_points - gt_center).abs() / half_size
        center_mask = center_delta.max(dim=1).values <= center_sampling_radius
        candidate_mask = inside & center_mask
        if not candidate_mask.any():
            candidate_mask = inside
        weights[candidate_mask] = torch.maximum(
            weights[candidate_mask],
            weights.new_full((int(candidate_mask.sum().item()),), float(box_weight.item())),
        )
    return weights


def max_prompt_logit_margin_loss(logits: torch.Tensor, margin: float) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    return F.relu(logits.max(dim=-1).values - margin)


def oversize_box_penalty(boxes: torch.Tensor, image_size: int, area_threshold: float) -> torch.Tensor:
    x1 = boxes[:, 0].clamp(0.0, float(image_size))
    y1 = boxes[:, 1].clamp(0.0, float(image_size))
    x2 = boxes[:, 2].clamp(0.0, float(image_size))
    y2 = boxes[:, 3].clamp(0.0, float(image_size))
    widths = (x2 - x1).clamp(min=0.0)
    heights = (y2 - y1).clamp(min=0.0)
    area_ratio = (widths * heights) / max(float(image_size * image_size), 1.0)
    edge_touch = (
        (x1 <= 1.0).float()
        + (y1 <= 1.0).float()
        + (x2 >= image_size - 1.0).float()
        + (y2 >= image_size - 1.0).float()
    ) / 4.0
    return (area_ratio - area_threshold).clamp(min=0.0) * (1.0 + edge_touch)


class PromptTaskAlignedAssigner:
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, center_sampling_radius: float = 0.5):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_sampling_radius = center_sampling_radius

    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_slot_priors: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
        pred_targetness: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        valid_class_mask: torch.Tensor,
    ) -> AssignmentResult:
        num_points, num_classes = pred_scores.shape
        device = pred_scores.device
        target_scores = torch.zeros((num_points, num_classes), device=device)
        target_boxes = torch.zeros((num_points, 4), device=device)
        fg_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        matched_gt_indices = torch.full((num_points,), -1, dtype=torch.long, device=device)
        matched_labels = torch.full((num_points,), -1, dtype=torch.long, device=device)

        if gt_boxes.numel() == 0:
            return AssignmentResult(
                target_scores,
                target_boxes,
                fg_mask,
                matched_gt_indices,
                matched_labels,
                torch.zeros(num_points, dtype=torch.bool, device=device),
            )

        assignment_metric = torch.full((num_points,), -1.0, device=device)
        overlaps = torch.zeros(num_points, device=device)
        pred_obj = pred_objectness.sigmoid()
        pred_tgt = pred_targetness.sigmoid()

        for gt_idx, gt_box in enumerate(gt_boxes):
            class_idx = int(gt_labels[gt_idx].item())
            if class_idx < 0 or class_idx >= num_classes or not valid_class_mask[class_idx]:
                continue

            inside = (
                (anchor_points[:, 0] >= gt_box[0])
                & (anchor_points[:, 0] <= gt_box[2])
                & (anchor_points[:, 1] >= gt_box[1])
                & (anchor_points[:, 1] <= gt_box[3])
            )
            gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
            half_size = ((gt_box[2:] - gt_box[:2]) * 0.5).clamp(min=1.0)
            center_delta = (anchor_points - gt_center).abs() / half_size
            center_mask = center_delta.max(dim=1).values <= self.center_sampling_radius
            candidate_inds = torch.nonzero(inside & center_mask, as_tuple=False).squeeze(1)
            if candidate_inds.numel() == 0:
                candidate_inds = torch.nonzero(inside, as_tuple=False).squeeze(1)
            if candidate_inds.numel() == 0:
                continue

            candidate_boxes = pred_boxes[candidate_inds]
            ious = bbox_iou(candidate_boxes, gt_box.unsqueeze(0)).squeeze(-1)
            cls_scores = pred_scores[candidate_inds, class_idx].sigmoid()
            center_prior = center_delta[candidate_inds].pow(2).sum(dim=1).mul(-0.5).exp()
            prior_scores = pred_slot_priors[candidate_inds, class_idx].clamp(min=1e-4)
            quality = (
                pred_obj[candidate_inds]
                * pred_tgt[candidate_inds]
                * cls_scores
                * prior_scores
            ).clamp(min=0.0).pow(0.25)
            align = quality.pow(self.alpha) * ious.pow(self.beta) * center_prior

            topk = min(self.topk, candidate_inds.numel())
            top_align, top_pos = torch.topk(align, topk)
            selected = candidate_inds[top_pos]
            update = top_align > assignment_metric[selected]
            selected = selected[update]
            if selected.numel() == 0:
                continue

            fg_mask[selected] = True
            assignment_metric[selected] = top_align[update]
            matched_gt_indices[selected] = gt_idx
            matched_labels[selected] = class_idx
            overlaps[selected] = bbox_iou(pred_boxes[selected], gt_box.unsqueeze(0)).squeeze(-1)

        if fg_mask.any():
            target_boxes[fg_mask] = gt_boxes[matched_gt_indices[fg_mask]]
            target_scores[fg_mask, matched_labels[fg_mask]] = overlaps[fg_mask].clamp(min=0.1)
        return AssignmentResult(
            target_scores,
            target_boxes,
            fg_mask,
            matched_gt_indices,
            matched_labels,
            torch.zeros(num_points, dtype=torch.bool, device=device),
        )


class PromptOneToOneAssigner:
    def __init__(
        self,
        center_sampling_radius: float = 0.75,
        candidate_topk: int = 8,
        duplicate_radius: float = 1.25,
    ):
        self.center_sampling_radius = center_sampling_radius
        self.candidate_topk = candidate_topk
        self.duplicate_radius = duplicate_radius

    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_slot_priors: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
        pred_targetness: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        valid_class_mask: torch.Tensor,
    ) -> AssignmentResult:
        num_points, num_classes = pred_scores.shape
        device = pred_scores.device
        target_scores = torch.zeros((num_points, num_classes), device=device)
        target_boxes = torch.zeros((num_points, 4), device=device)
        fg_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        matched_gt_indices = torch.full((num_points,), -1, dtype=torch.long, device=device)
        matched_labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
        duplicate_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

        if gt_boxes.numel() == 0:
            return AssignmentResult(target_scores, target_boxes, fg_mask, matched_gt_indices, matched_labels, duplicate_mask)

        pred_obj = pred_objectness.sigmoid()
        pred_tgt = pred_targetness.sigmoid()
        duplicate_candidates: Dict[int, torch.Tensor] = {}
        pair_candidates: List[tuple[float, int, int]] = []
        for gt_idx, gt_box in enumerate(gt_boxes):
            class_idx = int(gt_labels[gt_idx].item())
            if class_idx < 0 or class_idx >= num_classes or not valid_class_mask[class_idx]:
                continue

            inside = (
                (anchor_points[:, 0] >= gt_box[0])
                & (anchor_points[:, 0] <= gt_box[2])
                & (anchor_points[:, 1] >= gt_box[1])
                & (anchor_points[:, 1] <= gt_box[3])
            )
            gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
            half_size = ((gt_box[2:] - gt_box[:2]) * 0.5).clamp(min=1.0)
            center_delta = (anchor_points - gt_center).abs() / half_size
            center_mask = center_delta.max(dim=1).values <= self.center_sampling_radius
            candidate_inds = torch.nonzero(inside & center_mask, as_tuple=False).squeeze(1)
            if candidate_inds.numel() == 0:
                candidate_inds = torch.nonzero(inside, as_tuple=False).squeeze(1)
            if candidate_inds.numel() == 0:
                continue

            candidate_boxes = pred_boxes[candidate_inds]
            ious = bbox_iou(candidate_boxes, gt_box.unsqueeze(0)).squeeze(-1)
            cls_scores = pred_scores[candidate_inds, class_idx].sigmoid()
            prior_scores = pred_slot_priors[candidate_inds, class_idx].clamp(min=1e-4)
            quality = (
                pred_obj[candidate_inds]
                * pred_tgt[candidate_inds]
                * cls_scores
                * prior_scores
            ).clamp(min=0.0).pow(0.25)
            center_prior = center_delta[candidate_inds].pow(2).sum(dim=1).mul(-0.5).exp()
            metric = quality * ious.pow(2) * center_prior.pow(2)
            top_limit = min(candidate_inds.numel(), self.candidate_topk)
            top_metric, top_pos = torch.topk(metric, top_limit)
            for score, pos in zip(top_metric.tolist(), top_pos.tolist()):
                pair_candidates.append((score, int(candidate_inds[pos].item()), gt_idx))
            duplicate_center_mask = center_delta.max(dim=1).values <= self.duplicate_radius
            duplicate_candidates[gt_idx] = torch.nonzero(inside & duplicate_center_mask, as_tuple=False).squeeze(1)

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        used_preds = set()
        used_gts = set()
        overlaps = torch.zeros(num_points, device=device)
        for _, pred_idx, gt_idx in pair_candidates:
            if pred_idx in used_preds or gt_idx in used_gts:
                continue
            gt_box = gt_boxes[gt_idx]
            class_idx = int(gt_labels[gt_idx].item())
            fg_mask[pred_idx] = True
            matched_gt_indices[pred_idx] = gt_idx
            matched_labels[pred_idx] = class_idx
            overlaps[pred_idx] = bbox_iou(pred_boxes[pred_idx].unsqueeze(0), gt_box.unsqueeze(0)).squeeze()
            if gt_idx in duplicate_candidates:
                duplicate_mask[duplicate_candidates[gt_idx]] = True
                duplicate_mask[pred_idx] = False
            used_preds.add(pred_idx)
            used_gts.add(gt_idx)

        if fg_mask.any():
            target_boxes[fg_mask] = gt_boxes[matched_gt_indices[fg_mask]]
            target_scores[fg_mask, matched_labels[fg_mask]] = overlaps[fg_mask].clamp(min=0.1)
        duplicate_mask &= ~fg_mask
        return AssignmentResult(target_scores, target_boxes, fg_mask, matched_gt_indices, matched_labels, duplicate_mask)


class DFLoss(torch.nn.Module):
    def __init__(self, reg_max: int):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_left = target.long()
        target_right = (target_left + 1).clamp(max=self.reg_max - 1)
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist, target_left, reduction="none")
        loss_right = F.cross_entropy(pred_dist, target_right, reduction="none")
        return loss_left * weight_left + loss_right * weight_right


class PromptDetectionLoss(torch.nn.Module):
    def __init__(self, reg_max: int, cfg: LossConfig, context_cfg: ContextPainterConfig | None = None):
        super().__init__()
        self.reg_max = reg_max
        self.one2many_assigner = PromptTaskAlignedAssigner(
            cfg.tal_topk,
            cfg.tal_alpha,
            cfg.tal_beta,
            cfg.center_sampling_radius,
        )
        self.one2one_assigner = PromptOneToOneAssigner(
            cfg.one2one_center_sampling_radius,
            cfg.one2one_candidate_topk,
            cfg.one2one_duplicate_radius,
        )
        self.dfl = DFLoss(reg_max)
        self.cfg = cfg
        self.context_cfg = context_cfg or ContextPainterConfig(enabled=False)

    def _branch_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        assigner,
    ) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["pred_boxes"]
        pred_scores = outputs["pred_scores"]
        pred_slot_priors = outputs["pred_slot_priors"]
        pred_objectness = outputs["pred_objectness"]
        pred_targetness = outputs["pred_targetness"]
        pred_null_logits = outputs["pred_null_logits"]
        anchor_points = outputs["anchor_points"]
        stride_tensor = outputs["stride_tensor"]
        box_logits = outputs["box_distribution"]
        class_mask = outputs["class_mask"]

        total_objectness = pred_scores.new_tensor(0.0)
        total_targetness = pred_scores.new_tensor(0.0)
        total_null = pred_scores.new_tensor(0.0)
        total_match = pred_scores.new_tensor(0.0)
        total_iou = pred_scores.new_tensor(0.0)
        total_dfl = pred_scores.new_tensor(0.0)
        total_box_prior = pred_scores.new_tensor(0.0)
        total_pos = 0
        total_neg = 0
        total_pos_score_sum = pred_scores.new_tensor(0.0)
        total_neg_score_sum = pred_scores.new_tensor(0.0)
        total_matched_iou_sum = pred_scores.new_tensor(0.0)
        total_contrast = pred_scores.new_tensor(0.0)
        non_target_radius = getattr(assigner, "center_sampling_radius", self.cfg.non_target_center_sampling_radius)

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            non_target_boxes = target.get("non_target_boxes")
            non_target_weights = target.get("non_target_weights")
            valid_class_mask = class_mask[batch_idx]
            assign = assigner.assign(
                pred_scores[batch_idx],
                pred_slot_priors[batch_idx],
                pred_boxes[batch_idx],
                pred_objectness[batch_idx],
                pred_targetness[batch_idx],
                anchor_points,
                gt_boxes,
                gt_labels,
                valid_class_mask,
            )

            total_iou = total_iou + pred_boxes[batch_idx].sum() * 0.0
            total_dfl = total_dfl + box_logits[batch_idx].sum() * 0.0

            valid_logits = pred_scores[batch_idx][:, valid_class_mask]
            valid_targets = assign.target_scores[:, valid_class_mask]
            total_match = total_match + sigmoid_varifocal_loss(
                valid_logits,
                valid_targets,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="mean",
            )

            objectness_target = assign.target_scores.max(dim=-1).values
            total_objectness = total_objectness + sigmoid_varifocal_loss(
                pred_objectness[batch_idx],
                objectness_target,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="mean",
            )
            total_targetness = total_targetness + sigmoid_varifocal_loss(
                pred_targetness[batch_idx],
                objectness_target,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="mean",
            )
            null_target = (~assign.fg_mask).float()
            total_null = total_null + F.binary_cross_entropy_with_logits(
                pred_null_logits[batch_idx],
                null_target,
                reduction="mean",
            )

            pred_prob = pred_scores[batch_idx].sigmoid()
            pred_obj = pred_objectness[batch_idx].sigmoid()
            pred_tgt = pred_targetness[batch_idx].sigmoid()
            if assign.fg_mask.any():
                pos_boxes = pred_boxes[batch_idx][assign.fg_mask]
                tgt_boxes = assign.target_boxes[assign.fg_mask]
                iou = bbox_ciou(pos_boxes, tgt_boxes)
                total_iou = total_iou + (1 - iou).mean()
                total_matched_iou_sum = total_matched_iou_sum + bbox_iou(pos_boxes, tgt_boxes).sum()

                pos_labels = assign.matched_labels[assign.fg_mask]
                pos_joint_scores = torch.sqrt(
                    (pred_obj[assign.fg_mask] * pred_tgt[assign.fg_mask] * pred_prob[assign.fg_mask, pos_labels]).clamp(min=0.0)
                )
                total_pos_score_sum = total_pos_score_sum + pos_joint_scores.sum()

                cls_logits = pred_scores[batch_idx][assign.fg_mask]
                cls_logits = cls_logits.masked_fill(~valid_class_mask.unsqueeze(0), -1e4)
                if self.cfg.classification_margin > 0:
                    cls_logits = cls_logits.clone()
                    cls_logits[torch.arange(pos_labels.shape[0], device=pos_labels.device), pos_labels] -= self.cfg.classification_margin
                total_match = total_match + 0.5 * F.cross_entropy(cls_logits, pos_labels, reduction="mean")
                if valid_class_mask.sum() > 1:
                    contrast_logits = pred_scores[batch_idx][assign.fg_mask][:, valid_class_mask]
                    local_pos = pos_labels
                    pos_contrast_logits = contrast_logits[
                        torch.arange(local_pos.shape[0], device=local_pos.device),
                        local_pos,
                    ]
                    neg_contrast_logits = contrast_logits.clone()
                    neg_contrast_logits[torch.arange(local_pos.shape[0], device=local_pos.device), local_pos] = -1e4
                    hardest_negative = neg_contrast_logits.max(dim=-1).values
                    total_contrast = total_contrast + F.relu(
                        self.cfg.classification_margin - (pos_contrast_logits - hardest_negative)
                    ).mean()

                pos_logits = box_logits[batch_idx][assign.fg_mask]
                tgt_dist = bbox2dist(anchor_points[assign.fg_mask], tgt_boxes, self.reg_max, stride_tensor[assign.fg_mask])
                total_dfl = total_dfl + self.dfl(pos_logits.reshape(-1, self.reg_max), tgt_dist.reshape(-1)).mean()
                total_pos += int(assign.fg_mask.sum().item())

            if assign.duplicate_mask.any():
                dup_logits = pred_objectness[batch_idx][assign.duplicate_mask]
                dup_targets = torch.zeros_like(dup_logits)
                total_objectness = total_objectness + self.cfg.duplicate_weight * sigmoid_varifocal_loss(
                    dup_logits,
                    dup_targets,
                    alpha=self.cfg.focal_alpha,
                    gamma=self.cfg.focal_gamma,
                    reduction="mean",
                )
                dup_target_logits = pred_targetness[batch_idx][assign.duplicate_mask]
                total_targetness = total_targetness + self.cfg.duplicate_weight * sigmoid_varifocal_loss(
                    dup_target_logits,
                    dup_targets,
                    alpha=self.cfg.focal_alpha,
                    gamma=self.cfg.focal_gamma,
                    reduction="mean",
                )
                dup_scores = pred_scores[batch_idx][assign.duplicate_mask][:, valid_class_mask]
                if dup_scores.numel() > 0:
                    total_match = total_match + self.cfg.duplicate_weight * sigmoid_varifocal_loss(
                        dup_scores,
                        torch.zeros_like(dup_scores),
                        alpha=self.cfg.focal_alpha,
                        gamma=self.cfg.focal_gamma,
                        reduction="mean",
                    )
                    total_contrast = total_contrast + self.cfg.duplicate_weight * max_prompt_logit_margin_loss(
                        dup_scores,
                        margin=self.cfg.non_target_logit_margin,
                    ).mean()

            prompt_region_weights = build_box_region_weights(
                anchor_points,
                gt_boxes,
                gt_boxes.new_ones((gt_boxes.shape[0],)) if gt_boxes.numel() > 0 else gt_boxes.new_zeros((0,)),
                center_sampling_radius=non_target_radius,
            )
            non_target_region_weights = build_box_region_weights(
                anchor_points,
                non_target_boxes,
                non_target_weights,
                center_sampling_radius=self.cfg.non_target_center_sampling_radius,
            )
            non_target_region_weights = non_target_region_weights * self.cfg.non_target_weight
            non_target_mask = (non_target_region_weights > 0) & ~assign.fg_mask & ~assign.duplicate_mask
            non_target_mask &= prompt_region_weights <= 0
            if non_target_mask.any():
                region_weights = non_target_region_weights[non_target_mask]
                non_target_objectness = sigmoid_varifocal_loss(
                    pred_objectness[batch_idx][non_target_mask],
                    torch.zeros_like(pred_objectness[batch_idx][non_target_mask]),
                    alpha=self.cfg.focal_alpha,
                    gamma=self.cfg.focal_gamma,
                    reduction="none",
                )
                total_objectness = total_objectness + weighted_mean(non_target_objectness, region_weights)
                non_target_targetness = sigmoid_varifocal_loss(
                    pred_targetness[batch_idx][non_target_mask],
                    torch.zeros_like(pred_targetness[batch_idx][non_target_mask]),
                    alpha=self.cfg.focal_alpha,
                    gamma=self.cfg.focal_gamma,
                    reduction="none",
                )
                total_targetness = total_targetness + weighted_mean(non_target_targetness, region_weights)

                non_target_scores = pred_scores[batch_idx][non_target_mask][:, valid_class_mask]
                if non_target_scores.numel() > 0:
                    non_target_match = sigmoid_varifocal_loss(
                        non_target_scores,
                        torch.zeros_like(non_target_scores),
                        alpha=self.cfg.focal_alpha,
                        gamma=self.cfg.focal_gamma,
                        reduction="none",
                    )
                    total_match = total_match + weighted_mean(non_target_match, region_weights)
                    non_target_contrast = max_prompt_logit_margin_loss(
                        non_target_scores,
                        margin=self.cfg.non_target_logit_margin,
                    )
                    total_contrast = total_contrast + weighted_mean(non_target_contrast, region_weights)

            neg_mask = ~assign.fg_mask
            neg_count = int(neg_mask.sum().item())
            if neg_count > 0:
                total_neg += neg_count
                neg_scores = pred_prob[neg_mask][:, valid_class_mask]
                if neg_scores.numel() > 0:
                    neg_joint_scores = (
                        pred_obj[neg_mask]
                        * pred_tgt[neg_mask]
                        * neg_scores.max(dim=-1).values
                    ).clamp(min=0.0).pow(1.0 / 3.0)
                    total_neg_score_sum = total_neg_score_sum + neg_joint_scores.sum()

            oversize_mask = ~assign.fg_mask
            if oversize_mask.any():
                neg_scores_all = pred_prob[:, valid_class_mask]
                if neg_scores_all.numel() > 0:
                    prompt_keep = (1.0 - pred_null_logits[batch_idx].sigmoid()).clamp(min=0.0)
                    oversize_joint = (
                        pred_obj
                        * pred_tgt
                        * neg_scores_all.max(dim=-1).values
                        * prompt_keep
                    ).clamp(min=0.0).pow(0.25)
                    oversize_idx = torch.nonzero(oversize_mask, as_tuple=False).squeeze(1)
                    if oversize_idx.numel() > 0:
                        topk = min(self.cfg.oversize_box_topk, oversize_idx.numel())
                        _, order = torch.topk(oversize_joint[oversize_idx], k=topk)
                        oversize_idx = oversize_idx[order]
                        oversize_pen = oversize_box_penalty(
                            pred_boxes[batch_idx][oversize_idx],
                            image_size=target["image_size"],
                            area_threshold=self.cfg.oversize_box_threshold,
                        )
                        if oversize_pen.numel() > 0 and torch.any(oversize_pen > 0):
                            oversize_weights = oversize_joint[oversize_idx].detach().clamp(min=0.25)
                            total_box_prior = total_box_prior + self.cfg.oversize_box_weight * weighted_mean(
                                oversize_pen,
                                oversize_weights,
                            )

        num_batches = max(len(targets), 1)
        return {
            "loss_objectness": total_objectness / num_batches,
            "loss_targetness": total_targetness / num_batches,
            "loss_null": total_null / num_batches,
            "loss_match": total_match / num_batches,
            "loss_iou": total_iou / num_batches,
            "loss_dfl": total_dfl / num_batches,
            "loss_box_prior": total_box_prior / num_batches,
            "loss_contrast": total_contrast / num_batches,
            "num_pos": pred_scores.new_tensor(float(total_pos)),
            "num_neg": pred_scores.new_tensor(float(total_neg)),
            "mean_pos_score": (total_pos_score_sum / max(total_pos, 1)).detach(),
            "mean_neg_score": (total_neg_score_sum / max(total_neg, 1)).detach(),
            "mean_matched_iou": (total_matched_iou_sum / max(total_pos, 1)).detach(),
        }

    def _context_loss(
        self,
        context_aux: Dict[str, torch.Tensor] | None,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not self.context_cfg.enabled or context_aux is None:
            device = targets[0]["boxes"].device if targets else torch.device("cpu")
            zero = torch.zeros((), device=device)
            return {
                "loss_ctx_recon": zero,
                "loss_ctx_prior": zero,
            }

        pred_canvas = context_aux["query_canvas_pred_rgb"]
        slot_prior_logits = context_aux["slot_prior_logits"]
        total_recon = pred_canvas.new_tensor(0.0)
        total_prior = pred_canvas.new_tensor(0.0)

        for batch_idx, target in enumerate(targets):
            tgt_canvas = target["query_canvas_target"]
            tgt_label = target["query_canvas_label"]
            tgt_weight = target["query_canvas_weight"]

            recon = F.smooth_l1_loss(pred_canvas[batch_idx], tgt_canvas, reduction="none")
            recon_weight = tgt_weight.unsqueeze(0)
            total_recon = total_recon + (recon * recon_weight).sum() / recon_weight.sum().clamp(min=1.0)

            prior = F.cross_entropy(
                slot_prior_logits[batch_idx].unsqueeze(0),
                tgt_label.unsqueeze(0),
                reduction="none",
            ).squeeze(0)
            prior_weight = 1.0 + tgt_weight
            total_prior = total_prior + (prior * prior_weight).sum() / prior_weight.sum().clamp(min=1.0)

        num_batches = max(len(targets), 1)
        return {
            "loss_ctx_recon": total_recon / num_batches,
            "loss_ctx_prior": total_prior / num_batches,
        }

    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        one2many = self._branch_loss(outputs["one2many"], targets, self.one2many_assigner)
        one2one = self._branch_loss(outputs["one2one"], targets, self.one2one_assigner)
        context = self._context_loss(outputs.get("context_aux"), targets)

        loss_one2many = (
            self.cfg.objectness_weight * one2many["loss_objectness"]
            + self.cfg.targetness_weight * one2many["loss_targetness"]
            + self.cfg.null_weight * one2many["loss_null"]
            + self.cfg.match_weight * one2many["loss_match"]
            + self.cfg.iou_weight * one2many["loss_iou"]
            + self.cfg.dfl_weight * one2many["loss_dfl"]
            + one2many["loss_box_prior"]
            + self.cfg.contrast_weight * one2many["loss_contrast"]
        )
        loss_one2one = (
            self.cfg.objectness_weight * one2one["loss_objectness"]
            + self.cfg.targetness_weight * one2one["loss_targetness"]
            + self.cfg.null_weight * one2one["loss_null"]
            + self.cfg.match_weight * one2one["loss_match"]
            + self.cfg.iou_weight * one2one["loss_iou"]
            + self.cfg.dfl_weight * one2one["loss_dfl"]
            + one2one["loss_box_prior"]
            + self.cfg.contrast_weight * one2one["loss_contrast"]
        )
        total_loss = (
            self.cfg.one2many_weight * loss_one2many
            + self.cfg.one2one_weight * loss_one2one
            + self.context_cfg.recon_weight * context["loss_ctx_recon"]
            + self.context_cfg.prior_weight * context["loss_ctx_prior"]
        )

        return {
            "loss": total_loss,
            "loss_one2many": loss_one2many.detach(),
            "loss_one2one": loss_one2one.detach(),
            "loss_objectness": one2one["loss_objectness"],
            "loss_targetness": one2one["loss_targetness"],
            "loss_null": one2one["loss_null"],
            "loss_match": one2one["loss_match"],
            "loss_iou": one2one["loss_iou"],
            "loss_dfl": one2one["loss_dfl"],
            "loss_box_prior": one2one["loss_box_prior"],
            "loss_contrast": one2one["loss_contrast"],
            "loss_ctx_recon": context["loss_ctx_recon"],
            "loss_ctx_prior": context["loss_ctx_prior"],
            "num_pos": one2one["num_pos"],
            "num_neg": one2one["num_neg"],
            "mean_pos_score": one2one["mean_pos_score"],
            "mean_neg_score": one2one["mean_neg_score"],
            "mean_matched_iou": one2one["mean_matched_iou"],
        }
