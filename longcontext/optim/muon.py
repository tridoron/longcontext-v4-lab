from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


def zeropower_via_newton_schulz_5(
    G: torch.Tensor,
    eps: float = 1e-7,
    ns_steps: int = 5,
) -> torch.Tensor:
    """5-step Hybrid Newton-Schulz 正交化更新。"""
    assert G.ndim == 2
    if not 1 <= ns_steps <= 5:
        raise ValueError(f"ns_steps 必须在 [1, 5] 内，当前为 {ns_steps}")
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm(p="fro") + eps)
    coeffs = [
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (2.0, -1.5, 0.5),
        (2.0, -1.5, 0.5),
    ]
    for a, b, c in coeffs[:ns_steps]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        update_scale: float = 0.2,
        eps: float = 1e-7,
    ) -> None:
        if not 1 <= ns_steps <= 5:
            raise ValueError(f"ns_steps 必须在 [1, 5] 内，当前为 {ns_steps}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            update_scale=update_scale,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            scale = group["update_scale"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon 仅支持二维矩阵参数")
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newton_schulz_5(update, eps=eps, ns_steps=ns_steps)
                update = update * math.sqrt(max(p.shape)) * scale
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss
