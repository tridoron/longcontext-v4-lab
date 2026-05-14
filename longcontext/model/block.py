from __future__ import annotations

import torch
from torch import nn

from longcontext.model.attention_csa import CSALiteAttention
from longcontext.model.attention_full import FullAttention
from longcontext.model.attention_hca import HCALiteAttention
from longcontext.model.attention_hybrid import HybridAttention
from longcontext.model.attention_swa import SlidingWindowAttention
from longcontext.model.config import LongContextConfig
from longcontext.model.ffn import SwiGLUFFN
from longcontext.model.mhc import StaticMHCUpdate
from longcontext.model.rmsnorm import RMSNorm


def build_attention(config: LongContextConfig, layer_id: int) -> nn.Module:
    attn_type = config.attention.type
    if attn_type == "full":
        return FullAttention(config, layer_id)
    if attn_type == "swa":
        return SlidingWindowAttention(config, layer_id)
    if attn_type == "csa":
        return CSALiteAttention(config, layer_id)
    if attn_type == "hca":
        return HCALiteAttention(config, layer_id)
    if attn_type == "hybrid":
        return HybridAttention(config, layer_id)
    raise ValueError(f"未知 attention 类型: {attn_type}")


class TransformerBlock(nn.Module):
    def __init__(self, config: LongContextConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = build_attention(config, layer_id)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff)

        self.use_mhc = config.mhc.enabled
        if self.use_mhc:
            self.mhc_attn = StaticMHCUpdate(config.mhc.n_hc, config.mhc.sinkhorn_iters)
            self.mhc_ffn = StaticMHCUpdate(config.mhc.n_hc, config.mhc.sinkhorn_iters)

    def _attn_fn(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        return self.attn(self.attn_norm(x), attention_mask=attention_mask)

    def _ffn_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ffn_norm(x))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_mhc:
            x = x + self._attn_fn(x, attention_mask)
            x = x + self._ffn_fn(x)
            return x
        x = self.mhc_attn(x, lambda hidden: self._attn_fn(hidden, attention_mask))
        return self.mhc_ffn(x, self._ffn_fn)
