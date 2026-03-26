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


class PromptTaskAlignedAssigner:
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, center_sampling_radius: float = 0.5):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_sampling_radius = center_sampling_radius

    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
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
            quality = torch.sqrt((pred_obj[candidate_inds] * cls_scores).clamp(min=0.0))
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
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
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
            quality = torch.sqrt((pred_obj[candidate_inds] * cls_scores).clamp(min=0.0))
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
    def __init__(self, reg_max: int, cfg: LossConfig):
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

    def _branch_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        assigner,
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
        total_pos = 0
        total_neg = 0
        total_pos_score_sum = pred_scores.new_tensor(0.0)
        total_neg_score_sum = pred_scores.new_tensor(0.0)
        total_matched_iou_sum = pred_scores.new_tensor(0.0)

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            valid_class_mask = class_mask[batch_idx]
            assign = assigner.assign(
                pred_scores[batch_idx],
                pred_boxes[batch_idx],
                pred_objectness[batch_idx],
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

            pred_prob = pred_scores[batch_idx].sigmoid()
            pred_obj = pred_objectness[batch_idx].sigmoid()
            if assign.fg_mask.any():
                pos_boxes = pred_boxes[batch_idx][assign.fg_mask]
                tgt_boxes = assign.target_boxes[assign.fg_mask]
                iou = bbox_ciou(pos_boxes, tgt_boxes)
                total_iou = total_iou + (1 - iou).mean()
                total_matched_iou_sum = total_matched_iou_sum + bbox_iou(pos_boxes, tgt_boxes).sum()

                pos_labels = assign.matched_labels[assign.fg_mask]
                pos_joint_scores = torch.sqrt(
                    (pred_obj[assign.fg_mask] * pred_prob[assign.fg_mask, pos_labels]).clamp(min=0.0)
                )
                total_pos_score_sum = total_pos_score_sum + pos_joint_scores.sum()

                cls_logits = pred_scores[batch_idx][assign.fg_mask]
                cls_logits = cls_logits.masked_fill(~valid_class_mask.unsqueeze(0), -1e4)
                if self.cfg.classification_margin > 0:
                    cls_logits = cls_logits.clone()
                    cls_logits[torch.arange(pos_labels.shape[0], device=pos_labels.device), pos_labels] -= self.cfg.classification_margin
                total_match = total_match + 0.5 * F.cross_entropy(cls_logits, pos_labels, reduction="mean")

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
                dup_scores = pred_scores[batch_idx][assign.duplicate_mask][:, valid_class_mask]
                if dup_scores.numel() > 0:
                    total_match = total_match + self.cfg.duplicate_weight * sigmoid_varifocal_loss(
                        dup_scores,
                        torch.zeros_like(dup_scores),
                        alpha=self.cfg.focal_alpha,
                        gamma=self.cfg.focal_gamma,
                        reduction="mean",
                    )

            neg_mask = ~assign.fg_mask
            neg_count = int(neg_mask.sum().item())
            if neg_count > 0:
                total_neg += neg_count
                neg_scores = pred_prob[neg_mask][:, valid_class_mask]
                if neg_scores.numel() > 0:
                    neg_joint_scores = torch.sqrt((pred_obj[neg_mask] * neg_scores.max(dim=-1).values).clamp(min=0.0))
                    total_neg_score_sum = total_neg_score_sum + neg_joint_scores.sum()

        num_batches = max(len(targets), 1)
        return {
            "loss_objectness": total_objectness / num_batches,
            "loss_match": total_match / num_batches,
            "loss_iou": total_iou / num_batches,
            "loss_dfl": total_dfl / num_batches,
            "num_pos": pred_scores.new_tensor(float(total_pos)),
            "num_neg": pred_scores.new_tensor(float(total_neg)),
            "mean_pos_score": (total_pos_score_sum / max(total_pos, 1)).detach(),
            "mean_neg_score": (total_neg_score_sum / max(total_neg, 1)).detach(),
            "mean_matched_iou": (total_matched_iou_sum / max(total_pos, 1)).detach(),
        }

    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        one2many = self._branch_loss(outputs["one2many"], targets, self.one2many_assigner)
        one2one = self._branch_loss(outputs["one2one"], targets, self.one2one_assigner)

        loss_one2many = (
            self.cfg.objectness_weight * one2many["loss_objectness"]
            + self.cfg.match_weight * one2many["loss_match"]
            + self.cfg.iou_weight * one2many["loss_iou"]
            + self.cfg.dfl_weight * one2many["loss_dfl"]
        )
        loss_one2one = (
            self.cfg.objectness_weight * one2one["loss_objectness"]
            + self.cfg.match_weight * one2one["loss_match"]
            + self.cfg.iou_weight * one2one["loss_iou"]
            + self.cfg.dfl_weight * one2one["loss_dfl"]
        )
        total_loss = self.cfg.one2many_weight * loss_one2many + self.cfg.one2one_weight * loss_one2one

        return {
            "loss": total_loss,
            "loss_one2many": loss_one2many.detach(),
            "loss_one2one": loss_one2one.detach(),
            "loss_objectness": one2one["loss_objectness"],
            "loss_match": one2one["loss_match"],
            "loss_iou": one2one["loss_iou"],
            "loss_dfl": one2one["loss_dfl"],
            "loss_contrast": total_loss.new_tensor(0.0),
            "num_pos": one2one["num_pos"],
            "num_neg": one2one["num_neg"],
            "mean_pos_score": one2one["mean_pos_score"],
            "mean_neg_score": one2one["mean_neg_score"],
            "mean_matched_iou": one2one["mean_matched_iou"],
        }
