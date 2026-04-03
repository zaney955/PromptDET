from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    image_size: int = 256
    num_classes: int = 8
    max_prompt_classes: int = 4
    backbone_widths: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    neck_channels: int = 192
    prompt_dim: int = 256
    reg_max: int = 16
    num_attention_heads: int = 8
    grounding_dim: int = 192
    grounding_depth: int = 4
    prompt_crop_size: int = 128
    label_dropout: float = 0.4
    logit_scale_init: float = 1.5
    max_logit_scale: float = 2.5
    detail_score_weight: float = 0.5


@dataclass
class DenseGroundingConfig:
    enabled: bool = True
    scale: str = "p3"
    dim: int = 256
    depth: int = 4
    num_heads: int = 8
    feature_ensemble_start: int = 2
    slot_memory_tokens: int = 4
    recon_decoder_dim: int = 64
    query_mask_ratio: float = 1.0
    canvas_loss_weight: float = 1.0
    prior_weight: float = 1.0
    prior_warmup_ratio: float = 0.1
    random_color_min_distance: float = 0.45
    hint_inner_shrink: float = 0.6
    hint_bg_expand: float = 0.12
    prior_threshold: float = 0.15
    slot_loss_weight: float = 1.5
    fg_bce_weight: float = 1.0
    fg_dice_weight: float = 1.0
    center_loss_weight: float = 1.0
    prior_consistency_weight: float = 0.5
    target_channels: int = 5


ContextPainterConfig = DenseGroundingConfig


@dataclass
class DataConfig:
    train_list: str = ""
    val_list: str = ""
    train_labels_dir: str = ""
    val_labels_dir: str = ""
    class_names_path: str = ""
    min_prompt_classes: int = 1
    max_prompt_classes: int = 3
    max_prompt_instances_per_class: int = 2
    max_prompt_images: int = 4
    negative_episode_ratio: float = 0.5
    hard_positive_ratio: float = 0.5
    positive_query_shortlist: int = 6
    episodes_per_epoch: int = 2000
    val_episodes: int = 300
    num_workers: int = 2


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/default"
    batch_size: int = 8
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True
    grad_clip: float = 5.0
    score_threshold: float = 0.15
    pre_score_topk: int = 256
    local_peak_kernel: int = 3
    oversize_box_threshold: float = 0.85
    oversize_box_gamma: float = 20.0
    max_detections: int = 100
    eval_interval: int = 1
    save_interval: int = 1
    resume: str = ""


@dataclass
class LossConfig:
    tal_topk: int = 9
    tal_alpha: float = 1.0
    tal_beta: float = 6.0
    center_sampling_radius: float = 0.5
    one2one_center_sampling_radius: float = 0.75
    one2one_candidate_topk: int = 8
    one2one_duplicate_radius: float = 1.25
    non_target_center_sampling_radius: float = 0.75
    objectness_weight: float = 1.0
    targetness_weight: float = 1.0
    null_weight: float = 0.5
    match_weight: float = 1.0
    iou_weight: float = 7.5
    dfl_weight: float = 1.5
    duplicate_weight: float = 0.5
    non_target_weight: float = 1.0
    non_target_logit_margin: float = 0.0
    prompt_null_margin: float = 0.35
    prompt_null_weight: float = 0.5
    joint_score_weight: float = 0.75
    oversize_box_weight: float = 0.5
    oversize_box_threshold: float = 0.85
    oversize_box_topk: int = 16
    contrast_weight: float = 0.25
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    classification_margin: float = 0.2
    center_target_sigma: float = 0.35
    null_neg_pos_ratio: float = 16.0
    null_min_negatives: int = 32
    one2many_weight: float = 1.0
    one2one_weight: float = 1.5


@dataclass
class PromptDetConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dense_grounding: DenseGroundingConfig = field(default_factory=DenseGroundingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def context_painter(self) -> DenseGroundingConfig:
        return self.dense_grounding


def _update_dataclass(instance: Any, payload: Dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None) -> PromptDetConfig:
    config = PromptDetConfig()
    if path is None:
        return config
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "context_painter" in data and "dense_grounding" not in data:
        data["dense_grounding"] = data.pop("context_painter")
    if "data" in data:
        data["data"] = dict(data["data"])
        if "negative_ratio" in data["data"] and "negative_episode_ratio" not in data["data"]:
            data["data"]["negative_episode_ratio"] = data["data"].pop("negative_ratio")
    if "train" in data:
        data["train"] = dict(data["train"])
        legacy_train_keys = {
            "conf_threshold": "score_threshold",
            "pre_nms_topk": "pre_score_topk",
            "one2one_peak_kernel": "local_peak_kernel",
            "max_det": "max_detections",
        }
        for old_key, new_key in legacy_train_keys.items():
            if old_key in data["train"] and new_key not in data["train"]:
                data["train"][new_key] = data["train"].pop(old_key)
        data["train"].pop("nms_iou_threshold", None)
        data["train"].pop("one2one_topk", None)
    return _update_dataclass(config, data)


def save_config(path: str | Path, config: PromptDetConfig) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
