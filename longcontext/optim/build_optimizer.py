from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from longcontext.optim.muon import Muon


@dataclass
class OptimizerParameterGroups:
    muon_params: list
    adamw_decay: list
    adamw_no_decay: list
    beta_no_decay: list
    muon_names: list[str]
    adamw_names: list[str]


def _is_no_decay_name(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    markers = ("norm", "bias", "a_raw", "b_raw", "c_raw", "a_final_raw")
    return any(marker in name.lower() for marker in markers)


def _is_beta_raw_name(name: str) -> bool:
    return name.endswith(".beta_raw") or name == "beta_raw"


def _is_adamw_decay_name(name: str) -> bool:
    return name.startswith("tok_embeddings.") or name.startswith("lm_head.")


def collect_parameter_groups(model: torch.nn.Module, use_muon: bool = True) -> OptimizerParameterGroups:
    muon_params = []
    adamw_decay = []
    adamw_no_decay = []
    beta_no_decay = []
    muon_names = []
    adamw_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_beta_raw_name(name):
            beta_no_decay.append(param)
            adamw_names.append(f"beta_no_decay\t{name}")
        elif _is_no_decay_name(name, param):
            adamw_no_decay.append(param)
            adamw_names.append(f"no_decay\t{name}")
        elif _is_adamw_decay_name(name):
            adamw_decay.append(param)
            adamw_names.append(f"decay\t{name}")
        elif use_muon and param.ndim == 2:
            muon_params.append(param)
            muon_names.append(name)
        elif param.ndim == 2:
            adamw_decay.append(param)
            adamw_names.append(f"decay\t{name}")
        else:
            adamw_no_decay.append(param)
            adamw_names.append(f"no_decay\t{name}")
    return OptimizerParameterGroups(
        muon_params=muon_params,
        adamw_decay=adamw_decay,
        adamw_no_decay=adamw_no_decay,
        beta_no_decay=beta_no_decay,
        muon_names=muon_names,
        adamw_names=adamw_names,
    )


def split_parameter_groups(
    model: torch.nn.Module, use_muon: bool = True
) -> tuple[list, list, list, list[str], list[str]]:
    groups = collect_parameter_groups(model, use_muon=use_muon)
    return (
        groups.muon_params,
        groups.adamw_decay,
        groups.adamw_no_decay + groups.beta_no_decay,
        groups.muon_names,
        groups.adamw_names,
    )


class HybridOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for i, opt in enumerate(self.optimizers):
            key = f"optimizer_{i}"
            if key in state:
                opt.load_state_dict(state[key])


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
) -> HybridOptimizer:
    cfg = cfg or {}
    muon_cfg = cfg.get("muon", {})
    adamw_cfg = cfg.get("adamw", {})
    lr = float(cfg.get("lr", 3e-4))
    use_muon = bool(cfg.get("use_muon", False))
    groups = collect_parameter_groups(model, use_muon=use_muon)

    optimizers: list[torch.optim.Optimizer] = []
    if groups.muon_params:
        optimizers.append(
            Muon(
                groups.muon_params,
                lr=lr,
                weight_decay=float(muon_cfg.get("weight_decay", 0.1)),
                momentum=float(muon_cfg.get("momentum", 0.95)),
                nesterov=bool(muon_cfg.get("nesterov", True)),
                ns_steps=int(muon_cfg.get("ns_steps", 5)),
                update_scale=float(muon_cfg.get("update_scale", 0.2)),
                eps=float(muon_cfg.get("eps", 1e-7)),
            )
        )
    adamw_groups = []
    if groups.adamw_decay:
        adamw_groups.append(
            {
                "params": groups.adamw_decay,
                "weight_decay": float(adamw_cfg.get("weight_decay", 0.1)),
                "name": "adamw_decay",
            }
        )
    if groups.adamw_no_decay:
        adamw_groups.append(
            {
                "params": groups.adamw_no_decay,
                "weight_decay": float(adamw_cfg.get("no_decay_weight_decay", 0.0)),
                "name": "adamw_no_decay",
            }
        )
    if groups.beta_no_decay:
        adamw_groups.append({"params": groups.beta_no_decay, "weight_decay": 0.0, "name": "beta_raw"})
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(
                adamw_groups,
                lr=lr,
                betas=tuple(adamw_cfg.get("betas", [0.9, 0.95])),
                eps=float(adamw_cfg.get("eps", 1e-8)),
            )
        )
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "muon_param_names.txt").write_text(
            "\n".join(groups.muon_names) + "\n",
            encoding="utf-8",
        )
        (path / "adamw_param_names.txt").write_text(
            "\n".join(groups.adamw_names) + "\n",
            encoding="utf-8",
        )
    return HybridOptimizer(optimizers)
