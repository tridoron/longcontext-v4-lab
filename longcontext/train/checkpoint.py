from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from longcontext.model.config import LongContextConfig


def save_model_weights(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(tensors, str(path))


def load_model_weights(model: torch.nn.Module, path: str | Path, strict: bool = True) -> None:
    state = load_file(str(path))
    model.load_state_dict(state, strict=strict)


def save_config(config: LongContextConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any,
    step: int,
    seen_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "seen_tokens": seen_tokens,
            "extra": extra or {},
            "rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_training_state(path: str | Path, model: torch.nn.Module, optimizer: Any | None = None) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    return state
