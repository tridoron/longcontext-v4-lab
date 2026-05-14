from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def sinkhorn(x: torch.Tensor, iters: int = 10, eps: float = 1e-8) -> torch.Tensor:
    out = x
    for _ in range(iters):
        out = out / (out.sum(dim=-1, keepdim=True) + eps)
        out = out / (out.sum(dim=-2, keepdim=True) + eps)
    return out


class StaticMHCUpdate(nn.Module):
    def __init__(self, n_hc: int = 2, sinkhorn_iters: int = 10) -> None:
        super().__init__()
        self.n_hc = n_hc
        self.sinkhorn_iters = sinkhorn_iters
        self.a_raw = nn.Parameter(torch.zeros(1, n_hc))
        self.b_raw = nn.Parameter(torch.zeros(n_hc, n_hc))
        self.c_raw = nn.Parameter(torch.zeros(n_hc, 1))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.sigmoid(self.a_raw)
        b = sinkhorn(torch.exp(self.b_raw), self.sinkhorn_iters)
        c = 2.0 * torch.sigmoid(self.c_raw)
        return a, b, c

    def forward(self, streams: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # streams: [B, T, n_hc, D]
        a, b, c = self.coefficients()
        mixed_in = torch.einsum("sn,btsd->btd", a, streams)
        update = fn(mixed_in)
        carried = torch.einsum("ij,btjd->btid", b, streams)
        injected = c.view(1, 1, self.n_hc, 1) * update.unsqueeze(2)
        return carried + injected


class FinalMHCProjector(nn.Module):
    def __init__(self, n_hc: int = 2) -> None:
        super().__init__()
        if n_hc == 2:
            init = torch.tensor([2.0, 0.0])
        else:
            init = torch.zeros(n_hc)
            init[0] = 2.0
        self.a_final_raw = nn.Parameter(init)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        a_final = torch.softmax(self.a_final_raw, dim=-1)
        return torch.einsum("s,btsd->btd", a_final, streams)
