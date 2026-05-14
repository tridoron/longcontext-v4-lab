from __future__ import annotations

import math

import torch
from torch import nn

from longcontext.model.attention_utils import merge_heads, split_heads
from longcontext.model.config import LongContextConfig
from longcontext.model.rope import apply_rope, build_rope_freqs


class FullAttention(nn.Module):
    def __init__(self, config: LongContextConfig, layer_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        width = config.n_heads * config.d_head
        self.w_q = nn.Linear(config.d_model, width, bias=False)
        self.w_k = nn.Linear(config.d_model, width, bias=False)
        self.w_v = nn.Linear(config.d_model, width, bias=False)
        self.w_o = nn.Linear(width, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = split_heads(self.w_q(x), self.config.n_heads, self.config.d_head)
        k = split_heads(self.w_k(x), self.config.n_heads, self.config.d_head)
        v = split_heads(self.w_v(x), self.config.n_heads, self.config.d_head)
        cos, sin = build_rope_freqs(
            seq_len + position_offset,
            self.config.d_head,
            self.config.rope.base,
            self.config.rope.scaling_factor,
            x.device,
        )
        q = apply_rope(q, cos[position_offset:], sin[position_offset:])
        k = apply_rope(k, cos[position_offset:], sin[position_offset:])
        logits = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.config.d_head)
        causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~causal[None, None], float("-inf"))
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool)[:, None, None, :]
            logits = logits.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        return self.w_o(merge_heads(out))
