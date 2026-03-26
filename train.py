from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from promptdet.config import load_config, save_config
from promptdet.data.episodic import PromptEpisodeDataset, collate_episodes
from promptdet.engine.trainer import train
from promptdet.models.promptdet import PromptDET
from promptdet.utils.losses import PromptDetectionLoss
from promptdet.utils.misc import cleanup_distributed, is_main_process, set_seed, setup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptDET.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--train-annotations", type=str, default=None)
    parser.add_argument("--val-annotations", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-process batch size.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.train_annotations:
        config.data.train_annotations = args.train_annotations
    if args.val_annotations:
        config.data.val_annotations = args.val_annotations
    if args.images_dir:
        config.data.images_dir = args.images_dir
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.device:
        config.train.device = args.device

    if not config.data.train_annotations or not config.data.val_annotations or not config.data.images_dir:
        raise ValueError("train/val annotations and images_dir must be provided.")
    if config.data.max_prompt_classes > config.model.max_prompt_classes:
        raise ValueError("data.max_prompt_classes cannot exceed model.max_prompt_classes.")

    dist_info = setup_distributed(config.train.device, local_rank=args.local_rank)
    device = dist_info["device"]
    if device.type == "cuda":
        config.train.device = str(device)
    else:
        config.train.device = "cpu"

    set_seed(config.train.seed + dist_info["rank"])
    output_dir = Path(config.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(output_dir / "config.json", config)

    train_dataset = PromptEpisodeDataset(
        annotations_path=config.data.train_annotations,
        images_dir=config.data.images_dir,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.episodes_per_epoch,
        negative_ratio=config.data.negative_ratio,
        hard_negative_ratio=config.data.hard_negative_ratio,
        min_prompt_classes=config.data.min_prompt_classes,
        max_prompt_classes=config.data.max_prompt_classes,
        max_prompt_instances_per_class=config.data.max_prompt_instances_per_class,
        max_prompt_images=config.data.max_prompt_images,
    )
    val_dataset = PromptEpisodeDataset(
        annotations_path=config.data.val_annotations,
        images_dir=config.data.images_dir,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.val_episodes,
        negative_ratio=config.data.negative_ratio,
        hard_negative_ratio=config.data.hard_negative_ratio,
        min_prompt_classes=config.data.min_prompt_classes,
        max_prompt_classes=config.data.max_prompt_classes,
        max_prompt_instances_per_class=config.data.max_prompt_instances_per_class,
        max_prompt_images=config.data.max_prompt_images,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_info["distributed"] else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_info["distributed"] else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_episodes,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_episodes,
        pin_memory=device.type == "cuda",
    )

    model = PromptDET(config.model).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if dist_info["distributed"]:
        ddp_kwargs = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [dist_info["local_rank"]], "output_device": dist_info["local_rank"]})
        model = DDP(model, **ddp_kwargs)
    loss_fn = PromptDetectionLoss(config.model.reg_max, config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    try:
        result = train(model, loss_fn, train_loader, val_loader, optimizer, config)
        if is_main_process():
            print(result)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
