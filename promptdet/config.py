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
    prompt_crop_size: int = 128
    label_dropout: float = 0.4
    logit_scale_init: float = 1.5
    max_logit_scale: float = 2.5
    detail_score_weight: float = 0.5
    prompt_types: List[str] = field(default_factory=lambda: ["same_category", "same_instance", "part", "defect"])


@dataclass
class DataConfig:
    train_annotations: str = ""
    val_annotations: str = ""
    images_dir: str = ""
    category_map: str = ""
    min_prompt_classes: int = 1
    max_prompt_classes: int = 3
    max_prompt_instances_per_class: int = 2
    max_prompt_images: int = 4
    negative_ratio: float = 0.35
    hard_negative_ratio: float = 0.35
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
    conf_threshold: float = 0.15
    nms_iou_threshold: float = 0.5
    pre_nms_topk: int = 256
    one2one_topk: int = 300
    one2one_peak_kernel: int = 3
    class_margin_scale: float = 6.0
    max_det: int = 100
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
    match_weight: float = 1.0
    iou_weight: float = 7.5
    dfl_weight: float = 1.5
    duplicate_weight: float = 0.5
    non_target_weight: float = 1.0
    confusable_non_target_weight: float = 2.0
    non_target_logit_margin: float = 0.0
    contrast_weight: float = 0.25
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    classification_margin: float = 0.2
    one2many_weight: float = 1.0
    one2one_weight: float = 1.5


@dataclass
class PromptDetConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, payload: Dict[str, Any]) -> Any:
    for key, value in payload.items():
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
    return _update_dataclass(config, data)


def save_config(path: str | Path, config: PromptDetConfig) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
