from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from promptdet.config import PromptDetConfig, load_config, save_config
from promptdet.data.episodic import PromptEpisodeDataset, collate_episodes
from promptdet.engine.trainer import train
from promptdet.models.promptdet import PromptDET
from promptdet.utils.losses import PromptDetectionLoss
from promptdet.utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptDET.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--train-annotations", type=str, default=None)
    parser.add_argument("--val-annotations", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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

    set_seed(config.train.seed)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "config.json", config)

    train_dataset = PromptEpisodeDataset(
        annotations_path=config.data.train_annotations,
        images_dir=config.data.images_dir,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.episodes_per_epoch,
        negative_ratio=config.data.negative_ratio,
        hard_negative_ratio=config.data.hard_negative_ratio,
    )
    val_dataset = PromptEpisodeDataset(
        annotations_path=config.data.val_annotations,
        images_dir=config.data.images_dir,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.val_episodes,
        negative_ratio=config.data.negative_ratio,
        hard_negative_ratio=config.data.hard_negative_ratio,
    )
    model = PromptDET(config.model)
    loss_fn = PromptDetectionLoss(config.model.reg_max, config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True,
    )
    result = train(model, loss_fn, train_loader, val_loader, optimizer, config)
    print(result)


if __name__ == "__main__":
    main()
