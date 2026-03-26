from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from promptdet.config import LossConfig

from .box_ops import bbox2dist, bbox_ciou, bbox_iou


@dataclass
class AssignmentResult:
    target_scores: torch.Tensor
    target_boxes: torch.Tensor
    fg_mask: torch.Tensor
    matched_gt_indices: torch.Tensor
    matched_labels: torch.Tensor


class PromptTaskAlignedAssigner:
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
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
            return AssignmentResult(target_scores, target_boxes, fg_mask, matched_gt_indices, matched_labels)

        assignment_metric = torch.full((num_points,), -1.0, device=device)
        overlaps = torch.zeros(num_points, device=device)

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
            candidate_inds = torch.nonzero(inside, as_tuple=False).squeeze(1)
            if candidate_inds.numel() == 0:
                continue

            candidate_boxes = pred_boxes[candidate_inds]
            ious = bbox_iou(candidate_boxes, gt_box.unsqueeze(0)).squeeze(-1)
            cls_scores = pred_scores[candidate_inds, class_idx].sigmoid()
            align = cls_scores.pow(self.alpha) * ious.pow(self.beta)
            topk = min(self.topk, candidate_inds.numel())
            top_align, top_pos = torch.topk(align, topk)
            selected = candidate_inds[top_pos]
            update = top_align > assignment_metric[selected]
            selected = selected[update]
            top_align = top_align[update]
            if selected.numel() == 0:
                continue

            fg_mask[selected] = True
            assignment_metric[selected] = top_align
            matched_gt_indices[selected] = gt_idx
            matched_labels[selected] = class_idx
            overlaps[selected] = bbox_iou(pred_boxes[selected], gt_box.unsqueeze(0)).squeeze(-1)

        if fg_mask.any():
            matched_gt_boxes = gt_boxes[matched_gt_indices[fg_mask]]
            target_boxes[fg_mask] = matched_gt_boxes
            target_scores[fg_mask, matched_labels[fg_mask]] = overlaps[fg_mask].clamp(min=0.1)
        return AssignmentResult(target_scores, target_boxes, fg_mask, matched_gt_indices, matched_labels)


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


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    loss = ce * alpha_factor * (1 - p_t).pow(gamma)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class PromptDetectionLoss(torch.nn.Module):
    def __init__(self, reg_max: int, cfg: LossConfig):
        super().__init__()
        self.reg_max = reg_max
        self.assigner = PromptTaskAlignedAssigner(cfg.tal_topk, cfg.tal_alpha, cfg.tal_beta)
        self.dfl = DFLoss(reg_max)
        self.cfg = cfg

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        pred_boxes = outputs["pred_boxes"]
        pred_scores = outputs["pred_scores"]
        pred_objectness = outputs["pred_objectness"]
        anchor_points = outputs["anchor_points"]
        stride_tensor = outputs["stride_tensor"]
        box_logits = outputs["box_distribution"]
        class_mask = outputs["class_mask"]

        total_objectness = pred_scores.new_tensor(0.0)
        total_match = pred_scores.new_tensor(0.0)
        total_iou = pred_scores.new_tensor(0.0)
        total_dfl = pred_scores.new_tensor(0.0)
        total_contrast = pred_scores.new_tensor(0.0)
        total_pos = 0
        total_neg = 0
        total_pos_score_sum = pred_scores.new_tensor(0.0)
        total_neg_score_sum = pred_scores.new_tensor(0.0)
        total_matched_iou_sum = pred_scores.new_tensor(0.0)

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            valid_class_mask = class_mask[batch_idx]
            assign = self.assigner.assign(
                pred_scores[batch_idx],
                pred_boxes[batch_idx],
                anchor_points,
                gt_boxes,
                gt_labels,
                valid_class_mask,
            )

            objectness_target = assign.fg_mask.float()
            total_objectness = total_objectness + sigmoid_focal_loss(
                pred_objectness[batch_idx],
                objectness_target,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="mean",
            )

            pred_prob = pred_scores[batch_idx].sigmoid()
            pred_obj = pred_objectness[batch_idx].sigmoid()
            if assign.fg_mask.any():
                pos_boxes = pred_boxes[batch_idx][assign.fg_mask]
                tgt_boxes = assign.target_boxes[assign.fg_mask]
                iou = bbox_ciou(pos_boxes, tgt_boxes)
                total_iou = total_iou + (1 - iou).mean()
                total_matched_iou_sum = total_matched_iou_sum + bbox_iou(pos_boxes, tgt_boxes).sum()

                pos_labels = assign.matched_labels[assign.fg_mask]
                pos_joint_scores = pred_obj[assign.fg_mask] * pred_prob[assign.fg_mask, pos_labels]
                total_pos_score_sum = total_pos_score_sum + pos_joint_scores.sum()

                cls_logits = pred_scores[batch_idx][assign.fg_mask]
                cls_logits = cls_logits.masked_fill(~valid_class_mask.unsqueeze(0), -1e4)
                total_match = total_match + F.cross_entropy(cls_logits, pos_labels, reduction="mean")

                pos_logits = box_logits[batch_idx][assign.fg_mask]
                tgt_dist = bbox2dist(anchor_points[assign.fg_mask], tgt_boxes, self.reg_max, stride_tensor[assign.fg_mask])
                dfl_loss = self.dfl(
                    pos_logits.reshape(-1, self.reg_max),
                    tgt_dist.reshape(-1),
                )
                total_dfl = total_dfl + dfl_loss.mean()
                total_pos += int(assign.fg_mask.sum().item())

            neg_mask = ~assign.fg_mask
            neg_count = int(neg_mask.sum().item())
            if neg_count > 0:
                total_neg += neg_count
                neg_scores = pred_prob[neg_mask][:, valid_class_mask]
                if neg_scores.numel() > 0:
                    neg_joint_scores = pred_obj[neg_mask] * neg_scores.max(dim=-1).values
                    total_neg_score_sum = total_neg_score_sum + neg_joint_scores.sum()

        num_batches = max(len(targets), 1)
        mean_pos_score = total_pos_score_sum / max(total_pos, 1)
        mean_neg_score = total_neg_score_sum / max(total_neg, 1)
        mean_matched_iou = total_matched_iou_sum / max(total_pos, 1)
        total_loss = (
            self.cfg.objectness_weight * total_objectness
            + self.cfg.match_weight * total_match
            + self.cfg.iou_weight * total_iou
            + self.cfg.dfl_weight * total_dfl
            + self.cfg.contrast_weight * total_contrast
        ) / num_batches
        return {
            "loss": total_loss,
            "loss_objectness": total_objectness / num_batches,
            "loss_match": total_match / num_batches,
            "loss_iou": total_iou / num_batches,
            "loss_dfl": total_dfl / num_batches,
            "loss_contrast": total_contrast / num_batches,
            "num_pos": pred_scores.new_tensor(float(total_pos)),
            "num_neg": pred_scores.new_tensor(float(total_neg)),
            "mean_pos_score": mean_pos_score.detach(),
            "mean_neg_score": mean_neg_score.detach(),
            "mean_matched_iou": mean_matched_iou.detach(),
        }
