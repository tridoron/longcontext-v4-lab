from __future__ import annotations

import math

import torch
from torch import nn

from longcontext.model.attention_utils import (
    apply_block_rope,
    compress_kv,
    gather_local_kv,
    merge_heads,
    split_heads,
)
from longcontext.model.config import LongContextConfig
from longcontext.model.rope import apply_rope, build_rope_freqs


class HCALiteAttention(nn.Module):
    def __init__(
        self,
        config: LongContextConfig,
        layer_id: int = 0,
        compression_block_size: int | None = None,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hca = config.attention.hca
        self.block_size = compression_block_size or int(hca.get("compression_block_size", 64))
        self.local_window = local_window or int(hca.get("local_window", config.attention.local_window))
        width = config.n_heads * config.d_head
        self.w_q = nn.Linear(config.d_model, width, bias=False)
        self.w_k = nn.Linear(config.d_model, width, bias=False)
        self.w_v = nn.Linear(config.d_model, width, bias=False)
        self.w_o = nn.Linear(width, config.d_model, bias=False)
        self.w_z = nn.Linear(config.d_model, config.n_heads, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = split_heads(self.w_q(x), self.config.n_heads, self.config.d_head)
        k_raw = split_heads(self.w_k(x), self.config.n_heads, self.config.d_head)
        v_raw = split_heads(self.w_v(x), self.config.n_heads, self.config.d_head)
        cos, sin = build_rope_freqs(
            seq_len + position_offset,
            self.config.d_head,
            self.config.rope.base,
            self.config.rope.scaling_factor,
            x.device,
        )
        q = apply_rope(q, cos[position_offset:], sin[position_offset:])
        k_local = apply_rope(k_raw, cos[position_offset:], sin[position_offset:])
        z = self.w_z(x).view(bsz, seq_len, self.config.n_heads, 1)
        k_comp, v_comp = compress_kv(k_raw, v_raw, z, self.block_size)
        k_comp = apply_block_rope(
            k_comp,
            self.block_size,
            self.config.rope.base,
            self.config.rope.scaling_factor,
        )
        num_blocks = k_comp.shape[1]
        comp_shape = (bsz, seq_len, num_blocks, self.config.n_heads, self.config.d_head)
        k_comp_mem = k_comp[:, None].expand(
            bsz, seq_len, num_blocks, self.config.n_heads, self.config.d_head
        ).permute(0, 1, 3, 2, 4)
        v_comp_mem = v_comp[:, None].expand(comp_shape).permute(0, 1, 3, 2, 4)
        token_ids = torch.arange(seq_len, device=x.device)
        block_ids = torch.arange(num_blocks, device=x.device)
        comp_valid = block_ids[None, :] < (token_ids[:, None] // self.block_size)
        comp_logits = torch.einsum("bthd,bthgd->bthg", q, k_comp_mem) / math.sqrt(
            self.config.d_head
        )
        comp_logits = comp_logits.masked_fill(~comp_valid[None, :, None, :], torch.finfo(comp_logits.dtype).min)
        local_k, local_v, local_valid = gather_local_kv(k_local, v_raw, self.local_window, attention_mask)
        local_logits = torch.einsum("bthd,bthwd->bthw", q, local_k) / math.sqrt(self.config.d_head)
        local_logits = local_logits.masked_fill(~local_valid, torch.finfo(local_logits.dtype).min)
        logits = torch.cat((comp_logits, local_logits), dim=-1)
        values = torch.cat((v_comp_mem, local_v), dim=3)
        attn = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        out = torch.einsum("bthm,bthmd->bthd", attn, values)
        return self.w_o(merge_heads(out))
