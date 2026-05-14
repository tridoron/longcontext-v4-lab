from __future__ import annotations

import torch
from torch import nn

from longcontext.model.attention_utils import causal_local_attention, split_heads
from longcontext.model.config import LongContextConfig
from longcontext.model.rope import apply_rope, build_rope_freqs


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        config: LongContextConfig,
        layer_id: int = 0,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.local_window = local_window or config.attention.local_window
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
        seq_len = x.shape[1]
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
        return causal_local_attention(q, k, v, self.w_o, self.local_window, attention_mask)
