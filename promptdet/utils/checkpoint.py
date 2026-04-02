from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from promptdet.utils.misc import unwrap_model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_score: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    model_to_save = unwrap_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    incompatible = unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
    allowed_unexpected_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    unexpected_keys = [
        key for key in incompatible.unexpected_keys
        if not key.endswith(allowed_unexpected_suffixes)
    ]
    if incompatible.missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model. "
            f"Missing keys: {incompatible.missing_keys}. Unexpected keys: {unexpected_keys}."
        )
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
