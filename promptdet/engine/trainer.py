from __future__ import annotations

import math
from contextlib import nullcontext
import json
from pathlib import Path
import time
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
            "boxes": target["boxes"].to(device, non_blocking=True),
            "labels": target["labels"].to(device, non_blocking=True),
            "category_ids": target["category_ids"].to(device, non_blocking=True),
            "dense_slot_target": target["dense_slot_target"].to(device, non_blocking=True),
            "dense_fg_target": target["dense_fg_target"].to(device, non_blocking=True),
            "dense_center_target": target["dense_center_target"].to(device, non_blocking=True),
            "dense_valid_mask": target["dense_valid_mask"].to(device, non_blocking=True),
            "query_target_map": target["query_target_map"].to(device, non_blocking=True),
            "non_target_boxes": target["non_target_boxes"].to(device, non_blocking=True),
            "non_target_weights": target["non_target_weights"].to(device, non_blocking=True),
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
    amp_enabled = config.train.mixed_precision and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    scheduler = _make_scheduler(optimizer, config.train.epochs, config.train.warmup_epochs)
    start_epoch = 0
    best_f1 = 0.0
    history = []

    if config.train.resume:
        checkpoint = load_checkpoint(
            config.train.resume,
            model_without_ddp,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device.type,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_score", 0.0))
        history_path = output_dir / "history.json"
        if history_path.exists():
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            history = list(history_payload.get("history", []))
        else:
            extra = checkpoint.get("extra")
            if isinstance(extra, dict):
                history = [extra]
        if is_main_process():
            print(f"Resumed from {config.train.resume} at epoch {start_epoch} with best_f1={best_f1:.6f}")
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
            "loss_match": 0.0,
            "loss_iou": 0.0,
            "loss_dfl": 0.0,
            "loss_box_prior": 0.0,
            "loss_roi_contrast": 0.0,
            "loss_slot": 0.0,
            "loss_fg": 0.0,
            "loss_center": 0.0,
            "loss_prior_consistency": 0.0,
            "num_pos": 0.0,
            "num_neg": 0.0,
            "mean_pos_score": 0.0,
            "mean_neg_score": 0.0,
            "mean_matched_iou": 0.0,
            "time_data": 0.0,
            "time_h2d": 0.0,
            "time_forward": 0.0,
            "time_backward": 0.0,
            "time_optim": 0.0,
            "time_step": 0.0,
        }
        if hasattr(model_without_ddp, "set_context_prior_strength"):
            warmup_epochs = max(int(config.train.epochs * config.dense_grounding.prior_warmup_ratio), 1)
            if epoch + 1 <= warmup_epochs:
                ratio = (epoch + 1) / warmup_epochs
                model_without_ddp.set_context_prior_strength(0.25 + 0.75 * ratio)
            else:
                model_without_ddp.set_context_prior_strength(1.0)
        num_steps = 0
        step_end_time = time.perf_counter()
        for batch in pbar:
            data_time = time.perf_counter() - step_end_time
            h2d_start = time.perf_counter()
            prompt_images = batch["prompt_images"].to(device, non_blocking=True)
            prompt_image_mask = batch["prompt_image_mask"].to(device, non_blocking=True)
            prompt_boxes = batch["prompt_boxes"].to(device, non_blocking=True)
            prompt_hint_maps = batch["prompt_hint_maps"].to(device, non_blocking=True)
            prompt_target_maps = batch["prompt_target_maps"].to(device, non_blocking=True)
            prompt_crops = batch["prompt_crops"].to(device, non_blocking=True)
            prompt_class_indices = batch["prompt_class_indices"].to(device, non_blocking=True)
            prompt_source_indices = batch["prompt_source_indices"].to(device, non_blocking=True)
            prompt_instance_mask = batch["prompt_instance_mask"].to(device, non_blocking=True)
            prompt_class_mask = batch["prompt_class_mask"].to(device, non_blocking=True)
            query_image = batch["query_image"].to(device, non_blocking=True)
            query_target_map = batch["query_target_map"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            if device.type == "cuda" and config.train.debug_timing:
                torch.cuda.synchronize(device)
            h2d_time = time.perf_counter() - h2d_start

            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            forward_start = time.perf_counter()
            with amp_context:
                decoded = model(
                    prompt_images,
                    prompt_image_mask,
                    prompt_boxes,
                    prompt_hint_maps,
                    prompt_target_maps,
                    prompt_crops,
                    prompt_class_indices,
                    prompt_source_indices,
                    prompt_instance_mask,
                    prompt_class_mask,
                    query_image,
                    query_target_map=query_target_map,
                    decode=True,
                )
                losses = loss_fn(decoded, targets)
                loss = losses["loss"]
            if device.type == "cuda" and config.train.debug_timing:
                torch.cuda.synchronize(device)
            forward_time = time.perf_counter() - forward_start

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at epoch={epoch}, step={num_steps}.")

            backward_start = time.perf_counter()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if device.type == "cuda" and config.train.debug_timing:
                    torch.cuda.synchronize(device)
                backward_time = time.perf_counter() - backward_start
                optim_start = time.perf_counter()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip) if config.train.grad_clip > 0 else None
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm detected at epoch={epoch}, step={num_steps}.")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if device.type == "cuda" and config.train.debug_timing:
                    torch.cuda.synchronize(device)
                backward_time = time.perf_counter() - backward_start
                optim_start = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip) if config.train.grad_clip > 0 else None
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm detected at epoch={epoch}, step={num_steps}.")
                optimizer.step()
            if device.type == "cuda" and config.train.debug_timing:
                torch.cuda.synchronize(device)
            optim_time = time.perf_counter() - optim_start
            step_time = data_time + h2d_time + forward_time + backward_time + optim_time

            for key in epoch_stats:
                if key.startswith("time_"):
                    continue
                epoch_stats[key] += float(losses[key].item())
            epoch_stats["time_data"] += data_time
            epoch_stats["time_h2d"] += h2d_time
            epoch_stats["time_forward"] += forward_time
            epoch_stats["time_backward"] += backward_time
            epoch_stats["time_optim"] += optim_time
            epoch_stats["time_step"] += step_time
            num_steps += 1
            postfix = {
                "loss": f"{float(loss.item()):.4f}",
                "o2m": f"{float(losses['loss_one2many'].item()):.4f}",
                "o2o": f"{float(losses['loss_one2one'].item()):.4f}",
                "obj": f"{float(losses['loss_objectness'].item()):.4f}",
                "tgt": f"{float(losses['loss_targetness'].item()):.4f}",
                "cvs": f"{float(losses['loss_canvas'].item()):.4f}",
                "box": f"{float(losses['loss_box_prior'].item()):.4f}",
                "roi": f"{float(losses['loss_roi_contrast'].item()):.4f}",
                "grd": f"{float(losses['loss_slot'].item() + losses['loss_fg'].item() + losses['loss_center'].item()):.4f}",
                "pos": f"{float(losses['num_pos'].item()):.1f}",
                "ps": f"{float(losses['mean_pos_score'].item()):.3f}",
                "ns": f"{float(losses['mean_neg_score'].item()):.3f}",
            }
            if config.train.debug_timing:
                postfix.update({
                    "dt": f"{data_time:.2f}s",
                    "h2d": f"{h2d_time:.2f}s",
                    "fwd": f"{forward_time:.2f}s",
                    "bwd": f"{backward_time:.2f}s",
                    "opt": f"{optim_time:.2f}s",
                    "step": f"{step_time:.2f}s",
                })
            pbar.set_postfix(**postfix)
            step_end_time = time.perf_counter()

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
                save_checkpoint(
                    output_dir / "best.pt",
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_score=best_f1,
                    extra=summary,
                )
            best_f1 = max(best_f1, val_metrics["f1"])

        if is_main_process():
            history.append(summary)
            if (epoch + 1) % config.train.save_interval == 0:
                save_checkpoint(
                    output_dir / "last.pt",
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_score=best_f1,
                    extra=summary,
                )
            save_json(output_dir / "history.json", {"history": history, "best_f1": best_f1})

    return {"best_f1": best_f1}
