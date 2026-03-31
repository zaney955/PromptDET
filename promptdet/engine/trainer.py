from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from promptdet.config import PromptDetConfig
from promptdet.engine.evaluator import evaluate
from promptdet.utils.checkpoint import load_checkpoint, save_checkpoint
from promptdet.utils.misc import is_main_process, reduce_dict, save_json, unwrap_model


def _make_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_targets(targets, device: torch.device):
    moved = []
    for target in targets:
        moved.append({
            "boxes": target["boxes"].to(device),
            "labels": target["labels"].to(device),
            "category_ids": target["category_ids"].to(device),
            "dense_slot_target": target["dense_slot_target"].to(device),
            "dense_fg_target": target["dense_fg_target"].to(device),
            "dense_center_target": target["dense_center_target"].to(device),
            "dense_valid_mask": target["dense_valid_mask"].to(device),
            "query_target_map": target["query_target_map"].to(device),
            "non_target_boxes": target["non_target_boxes"].to(device),
            "non_target_weights": target["non_target_weights"].to(device),
            "image_size": target["image_size"],
        })
    return moved


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: PromptDetConfig,
) -> Dict[str, float]:
    public_metric_names = {
        "num_pos": "num_fg_anchors",
        "num_neg": "num_bg_anchors",
        "mean_pos_score": "mean_fg_score",
        "mean_neg_score": "mean_bg_score",
    }
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    model.to(device)
    model_without_ddp = unwrap_model(model)
    loss_fn.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.mixed_precision and device.type == "cuda")
    scheduler = _make_scheduler(optimizer, config.train.epochs, config.train.warmup_epochs)
    start_epoch = 0
    best_f1 = 0.0

    if config.train.resume:
        checkpoint = load_checkpoint(config.train.resume, model_without_ddp, optimizer=optimizer, scheduler=scheduler, map_location=device.type)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_score", 0.0))

    history = []
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.train.epochs}", leave=False, disable=not is_main_process())
        epoch_stats = {
            "loss": 0.0,
            "loss_one2many": 0.0,
            "loss_one2one": 0.0,
            "loss_canvas": 0.0,
            "loss_objectness": 0.0,
            "loss_targetness": 0.0,
            "loss_null": 0.0,
            "loss_match": 0.0,
            "loss_iou": 0.0,
            "loss_dfl": 0.0,
            "loss_box_prior": 0.0,
            "loss_contrast": 0.0,
            "loss_slot": 0.0,
            "loss_fg": 0.0,
            "loss_center": 0.0,
            "loss_prior_consistency": 0.0,
            "num_pos": 0.0,
            "num_neg": 0.0,
            "mean_pos_score": 0.0,
            "mean_neg_score": 0.0,
            "mean_matched_iou": 0.0,
        }
        if hasattr(model_without_ddp, "set_context_prior_strength"):
            warmup_epochs = max(int(config.train.epochs * config.dense_grounding.prior_warmup_ratio), 1)
            if epoch + 1 <= warmup_epochs:
                ratio = (epoch + 1) / warmup_epochs
                model_without_ddp.set_context_prior_strength(0.25 + 0.75 * ratio)
            else:
                model_without_ddp.set_context_prior_strength(1.0)
        num_steps = 0
        for batch in pbar:
            prompt_images = batch["prompt_images"].to(device)
            prompt_boxes = batch["prompt_boxes"].to(device)
            prompt_hint_maps = batch["prompt_hint_maps"].to(device)
            prompt_target_maps = batch["prompt_target_maps"].to(device)
            prompt_class_indices = batch["prompt_class_indices"].to(device)
            prompt_instance_mask = batch["prompt_instance_mask"].to(device)
            prompt_class_mask = batch["prompt_class_mask"].to(device)
            query_image = batch["query_image"].to(device)
            query_target_map = batch["query_target_map"].to(device)
            targets = _move_targets(batch["targets"], device)

            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled())
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                decoded = model(
                    prompt_images,
                    prompt_boxes,
                    prompt_hint_maps,
                    prompt_target_maps,
                    prompt_class_indices,
                    prompt_instance_mask,
                    prompt_class_mask,
                    query_image,
                    query_target_map=query_target_map,
                    decode=True,
                )
                losses = loss_fn(decoded, targets)
                loss = losses["loss"]

            scaler.scale(loss).backward()
            if config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key in epoch_stats:
                epoch_stats[key] += float(losses[key].item())
            num_steps += 1
            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                o2m=f"{float(losses['loss_one2many'].item()):.4f}",
                o2o=f"{float(losses['loss_one2one'].item()):.4f}",
                obj=f"{float(losses['loss_objectness'].item()):.4f}",
                tgt=f"{float(losses['loss_targetness'].item()):.4f}",
                nul=f"{float(losses['loss_null'].item()):.4f}",
                cvs=f"{float(losses['loss_canvas'].item()):.4f}",
                box=f"{float(losses['loss_box_prior'].item()):.4f}",
                grd=f"{float(losses['loss_slot'].item() + losses['loss_fg'].item() + losses['loss_center'].item()):.4f}",
                pos=f"{float(losses['num_pos'].item()):.1f}",
                ps=f"{float(losses['mean_pos_score'].item()):.3f}",
                ns=f"{float(losses['mean_neg_score'].item()):.3f}",
            )

        scheduler.step()
        reduced_epoch_stats = reduce_dict(epoch_stats, average=False)
        total_steps = reduce_dict({"num_steps": float(num_steps)}, average=False)["num_steps"]
        train_metrics = {
            public_metric_names.get(key, key): value / max(total_steps, 1.0)
            for key, value in reduced_epoch_stats.items()
        }
        summary = {"epoch": epoch, **train_metrics}

        if (epoch + 1) % config.train.eval_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                score_threshold=config.train.score_threshold,
                pre_score_topk=config.train.pre_score_topk,
                local_peak_kernel=config.train.local_peak_kernel,
                oversize_box_threshold=config.train.oversize_box_threshold,
                oversize_box_gamma=config.train.oversize_box_gamma,
                max_detections=config.train.max_detections,
            )
            summary.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["f1"] >= best_f1 and is_main_process():
                best_f1 = val_metrics["f1"]
                save_checkpoint(output_dir / "best.pt", model_without_ddp, optimizer, scheduler, epoch=epoch, best_score=best_f1, extra=summary)
            best_f1 = max(best_f1, val_metrics["f1"])

        if is_main_process():
            history.append(summary)
            if (epoch + 1) % config.train.save_interval == 0:
                save_checkpoint(output_dir / "last.pt", model_without_ddp, optimizer, scheduler, epoch=epoch, best_score=best_f1, extra=summary)
            save_json(output_dir / "history.json", {"history": history, "best_f1": best_f1})

    return {"best_f1": best_f1}
