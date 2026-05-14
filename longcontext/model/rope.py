from __future__ import annotations

import torch


def build_rope_freqs(
    seq_len: int,
    rope_dim: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    positions = positions / scaling_factor
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_rope_freqs_for_positions(
    positions: torch.Tensor,
    rope_dim: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = positions.to(dtype=torch.float32) / scaling_factor
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=positions.device, dtype=torch.float32) / rope_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D]；cos/sin 可为 [T, D/2] 或 [G, D/2]。
    rope_dim = cos.shape[-1] * 2
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    cos_full = torch.repeat_interleave(cos, 2, dim=-1).to(dtype=x.dtype)[None, :, None, :]
    sin_full = torch.repeat_interleave(sin, 2, dim=-1).to(dtype=x.dtype)[None, :, None, :]
    return torch.cat((x_rope * cos_full + rotate_half(x_rope) * sin_full, x_pass), dim=-1)
