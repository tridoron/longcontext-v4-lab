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


class CSALiteAttention(nn.Module):
    def __init__(
        self,
        config: LongContextConfig,
        layer_id: int = 0,
        compression_block_size: int | None = None,
        top_k: int | None = None,
        local_window: int | None = None,
        index_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        csa = config.attention.csa
        self.block_size = compression_block_size or int(csa.get("compression_block_size", 4))
        self.top_k = top_k or int(csa.get("top_k", config.attention.top_k))
        self.local_window = local_window or int(csa.get("local_window", config.attention.local_window))
        self.index_dim = index_dim or int(csa.get("index_dim", config.attention.index_dim))
        width = config.n_heads * config.d_head
        self.w_q = nn.Linear(config.d_model, width, bias=False)
        self.w_k = nn.Linear(config.d_model, width, bias=False)
        self.w_v = nn.Linear(config.d_model, width, bias=False)
        self.w_o = nn.Linear(width, config.d_model, bias=False)
        self.w_z = nn.Linear(config.d_model, config.n_heads, bias=False)
        self.w_iq = nn.Linear(config.d_model, self.index_dim, bias=False)
        self.w_ik = nn.Linear(width, self.index_dim, bias=False)
        self.beta_raw = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

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

        index_q = self.w_iq(x)
        index_k = self.w_ik(k_comp.reshape(bsz, num_blocks, -1))
        index_scores = torch.einsum("btd,bgd->btg", index_q, index_k) / math.sqrt(self.index_dim)
        token_ids = torch.arange(seq_len, device=x.device)
        block_ids = torch.arange(num_blocks, device=x.device)
        valid_block = block_ids[None, :] < (token_ids[:, None] // self.block_size)
        index_scores = index_scores.masked_fill(~valid_block[None], torch.finfo(index_scores.dtype).min)
        k_eff = min(self.top_k, num_blocks)
        topk_scores, topk_indices = torch.topk(index_scores, k=k_eff, dim=-1)
        selected_valid = torch.isfinite(topk_scores)

        batch_idx = torch.arange(bsz, device=x.device)[:, None, None]
        selected_k = k_comp[batch_idx, topk_indices].permute(0, 1, 3, 2, 4)
        selected_v = v_comp[batch_idx, topk_indices].permute(0, 1, 3, 2, 4)

        local_k, local_v, local_valid = gather_local_kv(k_local, v_raw, self.local_window, attention_mask)
        comp_logits = torch.einsum("bthd,bthkd->bthk", q, selected_k) / math.sqrt(
            self.config.d_head
        )
        beta = torch.sigmoid(self.beta_raw)
        comp_logits = comp_logits + beta * topk_scores[:, :, None, :]
        comp_valid = selected_valid[:, :, None, :]
        comp_logits = comp_logits.masked_fill(~comp_valid, torch.finfo(comp_logits.dtype).min)
        local_logits = torch.einsum("bthd,bthwd->bthw", q, local_k) / math.sqrt(self.config.d_head)
        local_logits = local_logits.masked_fill(~local_valid, torch.finfo(local_logits.dtype).min)

        logits = torch.cat((comp_logits, local_logits), dim=-1)
        values = torch.cat((selected_v, local_v), dim=3)
        attn = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        out = torch.einsum("bthm,bthmd->bthd", attn, values)
        return self.w_o(merge_heads(out))
