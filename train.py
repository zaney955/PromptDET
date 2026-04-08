from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from promptdet.config import load_config, save_config
from promptdet.data.episodic import PromptEpisodeDataset, collate_episodes
from promptdet.data.resize_cache import ensure_resize_cache
from promptdet.data.yolo_io import load_image_list
from promptdet.engine.trainer import train
from promptdet.models.promptdet import PromptDET
from promptdet.utils.losses import PromptDetectionLoss
from promptdet.utils.misc import cleanup_distributed, is_main_process, set_seed, setup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptDET.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from a checkpoint. Pass a path, or omit the value to use output_dir/last.pt.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    resume_arg = args.resume
    if config_path is None and resume_arg and resume_arg != "auto":
        resume_config = Path(resume_arg).expanduser().resolve().parent / "config.json"
        if resume_config.exists():
            config_path = str(resume_config)
        else:
            raise FileNotFoundError(
                f"--resume was provided but no --config was given, and {resume_config} does not exist."
            )
    if config_path is None and resume_arg == "auto":
        raise ValueError("--resume without a checkpoint path requires --config so output_dir can be resolved.")
    config = load_config(config_path)
    if resume_arg:
        if resume_arg == "auto":
            auto_resume = Path(config.train.output_dir).expanduser().resolve() / "last.pt"
            if not auto_resume.exists():
                raise FileNotFoundError(f"--resume requested automatic recovery, but {auto_resume} does not exist.")
            config.train.resume = str(auto_resume)
        else:
            config.train.resume = resume_arg
    if args.device:
        config.train.device = args.device

    if (
        not config.data.train_list
        or not config.data.val_list
        or not config.data.labels_dir
        or not config.data.class_names
    ):
        raise ValueError("train/val txt lists, labels_dir, and data.class_names must be provided.")
    if config.data.max_prompt_classes > config.model.max_prompt_classes:
        raise ValueError(
            "data.max_prompt_classes cannot exceed model.max_prompt_classes "
            f"(got data={config.data.max_prompt_classes}, model={config.model.max_prompt_classes})."
        )

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

    resize_cache_dir = None
    if config.data.resize_cache_enabled:
        resize_cache_dir = config.data.resize_cache_dir or str((output_dir / "resize_cache").resolve())
        if is_main_process():
            cache_image_paths = load_image_list(config.data.train_list) + load_image_list(config.data.val_list)
            cache_summary = ensure_resize_cache(
                image_paths=cache_image_paths,
                image_size=config.model.image_size,
                cache_dir=resize_cache_dir,
                num_workers=config.data.resize_cache_workers,
            )
            print(
                "Resize cache ready:",
                f"total={cache_summary['total']}",
                f"updated={cache_summary['updated']}",
                f"skipped={cache_summary['skipped']}",
                f"dir={resize_cache_dir}",
            )
        if dist_info["distributed"] and dist.is_initialized():
            if device.type == "cuda":
                dist.barrier(device_ids=[dist_info["local_rank"]])
            else:
                dist.barrier()
    prompt_crop_cache_dir = None
    if config.data.prompt_crop_cache_enabled:
        prompt_crop_cache_dir = config.data.prompt_crop_cache_dir or resize_cache_dir
        if prompt_crop_cache_dir is None:
            prompt_crop_cache_dir = str((output_dir / "prompt_crop_cache").resolve())

    train_dataset = PromptEpisodeDataset(
        image_list_path=config.data.train_list,
        labels_dir=config.data.labels_dir,
        class_names=config.data.class_names,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.episodes_per_epoch,
        negative_episode_ratio=config.data.negative_episode_ratio,
        min_prompt_classes=config.data.min_prompt_classes,
        max_prompt_classes=config.data.max_prompt_classes,
        max_prompt_instances_per_class=config.data.max_prompt_instances_per_class,
        max_prompt_images=config.data.max_prompt_images,
        prompt_crop_size=config.model.prompt_crop_size,
        color_min_distance=config.dense_grounding.random_color_min_distance,
        center_target_sigma=config.loss.center_target_sigma,
        hint_inner_shrink=config.dense_grounding.hint_inner_shrink,
        hint_bg_expand=config.dense_grounding.hint_bg_expand,
        hard_positive_ratio=config.data.hard_positive_ratio,
        positive_query_shortlist=config.data.positive_query_shortlist,
        resize_cache_dir=resize_cache_dir,
        prompt_crop_cache_dir=prompt_crop_cache_dir,
        seed=None,
    )
    val_dataset = PromptEpisodeDataset(
        image_list_path=config.data.val_list,
        labels_dir=config.data.labels_dir,
        class_names=config.data.class_names,
        image_size=config.model.image_size,
        episodes_per_epoch=config.data.val_episodes,
        negative_episode_ratio=config.data.negative_episode_ratio,
        min_prompt_classes=config.data.min_prompt_classes,
        max_prompt_classes=config.data.max_prompt_classes,
        max_prompt_instances_per_class=config.data.max_prompt_instances_per_class,
        max_prompt_images=config.data.max_prompt_images,
        prompt_crop_size=config.model.prompt_crop_size,
        color_min_distance=config.dense_grounding.random_color_min_distance,
        center_target_sigma=config.loss.center_target_sigma,
        hint_inner_shrink=config.dense_grounding.hint_inner_shrink,
        hint_bg_expand=config.dense_grounding.hint_bg_expand,
        hard_positive_ratio=config.data.hard_positive_ratio,
        positive_query_shortlist=config.data.positive_query_shortlist,
        resize_cache_dir=resize_cache_dir,
        prompt_crop_cache_dir=prompt_crop_cache_dir,
        seed=config.train.seed + 100_000,
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
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_episodes,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )

    model = PromptDET(config.model, config.dense_grounding)
    if hasattr(model, "set_activation_checkpointing"):
        model.set_activation_checkpointing(config.train.activation_checkpointing)
    model = model.to(device)
    if dist_info["distributed"]:
        ddp_kwargs = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [dist_info["local_rank"]], "output_device": dist_info["local_rank"]})
        model = DDP(model, **ddp_kwargs)
    loss_fn = PromptDetectionLoss(config.model.reg_max, config.loss, config.dense_grounding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    try:
        result = train(model, loss_fn, train_loader, val_loader, optimizer, config)
        if is_main_process():
            print(result)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
