from __future__ import annotations

import math

import torch
from torch import nn

from longcontext.model.config import LongContextConfig
from longcontext.model.rope import apply_rope, build_rope_freqs, build_rope_freqs_for_positions


def split_heads(x: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    return x.view(bsz, seq_len, n_heads, d_head)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, n_heads, d_head = x.shape
    return x.contiguous().view(bsz, seq_len, n_heads * d_head)


def local_positions(
    seq_len: int, window: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(seq_len, device=device)
    offsets = torch.arange(window, device=device)
    pos = token_ids[:, None] - (window - 1 - offsets)[None, :]
    valid = pos >= 0
    return pos.clamp_min(0), valid


def causal_compressed_block_mask(
    seq_len: int,
    num_blocks: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    token_ids = torch.arange(seq_len, device=device)
    block_ids = torch.arange(num_blocks, device=device)
    return block_ids[None, :] < (token_ids[:, None] // block_size)


def gather_local_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, d_head = k.shape
    pos, valid = local_positions(seq_len, window, k.device)
    flat_pos = pos.reshape(-1)
    local_k = k[:, flat_pos].view(bsz, seq_len, window, n_heads, d_head).permute(0, 1, 3, 2, 4)
    local_v = v[:, flat_pos].view(bsz, seq_len, window, n_heads, d_head).permute(0, 1, 3, 2, 4)
    local_valid = valid[None, :, None, :]
    if attention_mask is not None:
        key_valid = attention_mask.to(torch.bool)[:, flat_pos].view(bsz, seq_len, window)
        local_valid = local_valid & key_valid[:, :, None, :]
    return local_k, local_v, local_valid


def causal_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_proj: nn.Linear,
    window: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    local_k, local_v, local_valid = gather_local_kv(k, v, window, attention_mask)
    logits = torch.einsum("bthd,bthwd->bthw", q, local_k) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~local_valid, float("-inf"))
    attn = torch.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
    out = torch.einsum("bthw,bthwd->bthd", attn, local_v)
    return out_proj(merge_heads(out))


class QKVProjectionMixin:
    config: LongContextConfig
    w_q: nn.Linear
    w_k: nn.Linear
    w_v: nn.Linear
    w_o: nn.Linear

    def project_qkv(
        self, x: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = split_heads(self.w_q(x), self.config.n_heads, self.config.d_head)
        k = split_heads(self.w_k(x), self.config.n_heads, self.config.d_head)
        v = split_heads(self.w_v(x), self.config.n_heads, self.config.d_head)
        seq_len = x.shape[1]
        cos, sin = build_rope_freqs(
            seq_len + position_offset,
            self.config.d_head,
            self.config.rope.base,
            self.config.rope.scaling_factor,
            x.device,
        )
        q = apply_rope(q, cos[position_offset:], sin[position_offset:])
        k = apply_rope(k, cos[position_offset:], sin[position_offset:])
        return q, k, v


def apply_block_rope(
    k_comp: torch.Tensor,
    block_size: int,
    base: float,
    scaling_factor: float,
) -> torch.Tensor:
    num_blocks = k_comp.shape[1]
    positions = (torch.arange(num_blocks, device=k_comp.device) + 1) * block_size - 1
    cos, sin = build_rope_freqs_for_positions(positions, k_comp.shape[-1], base, scaling_factor)
    return apply_rope(k_comp, cos, sin)


def pad_to_block(x: torch.Tensor, block_size: int, fill_value: float = 0.0) -> torch.Tensor:
    seq_len = x.shape[1]
    pad_len = (block_size - seq_len % block_size) % block_size
    if pad_len == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = pad_len
    pad = torch.full(pad_shape, fill_value, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=1)


def compress_kv(
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    z: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, n_heads, d_head = k_raw.shape
    k_pad = pad_to_block(k_raw, block_size, 0.0)
    v_pad = pad_to_block(v_raw, block_size, 0.0)
    z_pad = pad_to_block(z, block_size, torch.finfo(z.dtype).min)
    num_blocks = k_pad.shape[1] // block_size
    k_blk = k_pad.view(bsz, num_blocks, block_size, n_heads, d_head)
    v_blk = v_pad.view(bsz, num_blocks, block_size, n_heads, d_head)
    z_blk = z_pad.view(bsz, num_blocks, block_size, n_heads, 1)
    alpha = torch.softmax(z_blk.float(), dim=2).to(dtype=k_raw.dtype)
    return (alpha * k_blk).sum(dim=2), (alpha * v_blk).sum(dim=2)
