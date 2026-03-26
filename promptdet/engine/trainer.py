from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from promptdet.config import PromptDetConfig
from promptdet.engine.evaluator import evaluate
from promptdet.utils.checkpoint import load_checkpoint, save_checkpoint
from promptdet.utils.misc import save_json


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
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu")
    model.to(device)
    loss_fn.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.mixed_precision and device.type == "cuda")
    scheduler = _make_scheduler(optimizer, config.train.epochs, config.train.warmup_epochs)
    start_epoch = 0
    best_f1 = 0.0

    if config.train.resume:
        checkpoint = load_checkpoint(config.train.resume, model, optimizer=optimizer, scheduler=scheduler, map_location=device.type)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_score", 0.0))

    history = []
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.train.epochs}", leave=False)
        epoch_stats = {
            "loss": 0.0,
            "loss_match": 0.0,
            "loss_iou": 0.0,
            "loss_dfl": 0.0,
            "loss_contrast": 0.0,
            "num_pos": 0.0,
            "num_neg": 0.0,
            "mean_pos_score": 0.0,
            "mean_neg_score": 0.0,
            "mean_matched_iou": 0.0,
        }
        num_steps = 0
        for batch in pbar:
            support_image = batch["support_image"].to(device)
            support_box = batch["support_box"].to(device)
            prompt_label = batch["prompt_label"].to(device)
            prompt_type = batch["prompt_type"].to(device)
            query_image = batch["query_image"].to(device)
            targets = _move_targets(batch["targets"], device)

            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled())
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                raw = model(support_image, support_box, prompt_label, query_image, prompt_type)
                decoded = model.decode_raw(raw)
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
                pos=f"{float(losses['num_pos'].item()):.1f}",
                ps=f"{float(losses['mean_pos_score'].item()):.3f}",
                ns=f"{float(losses['mean_neg_score'].item()):.3f}",
            )

        scheduler.step()
        train_metrics = {key: value / max(num_steps, 1) for key, value in epoch_stats.items()}
        summary = {"epoch": epoch, **train_metrics}

        if (epoch + 1) % config.train.eval_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                conf_threshold=config.train.conf_threshold,
                nms_iou_threshold=config.train.nms_iou_threshold,
                max_det=config.train.max_det,
            )
            summary.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["f1"] >= best_f1:
                best_f1 = val_metrics["f1"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch=epoch, best_score=best_f1, extra=summary)

        history.append(summary)
        if (epoch + 1) % config.train.save_interval == 0:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch=epoch, best_score=best_f1, extra=summary)
        save_json(output_dir / "history.json", {"history": history, "best_f1": best_f1})

    return {"best_f1": best_f1}
