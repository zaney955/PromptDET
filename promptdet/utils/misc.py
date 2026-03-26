from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_dict(values: Dict[str, float], average: bool = True) -> Dict[str, float]:
    if not is_dist_available_and_initialized():
        return values
    keys = sorted(values.keys())
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return {key: float(val) for key, val in zip(keys, tensor.tolist())}


def reduce_tensor(tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
    if not is_dist_available_and_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= get_world_size()
    return reduced


def setup_distributed(device: str, local_rank: int | None = None) -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_rank = env_local_rank if local_rank is None else local_rank
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch_device = torch.device(f"cuda:{local_rank}")
    else:
        torch_device = torch.device("cpu")

    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": torch_device,
    }


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()
