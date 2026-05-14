from __future__ import annotations

import math

import torch

from longcontext.model.attention_csa import CSALiteAttention
from longcontext.model.attention_full import FullAttention
from longcontext.model.attention_hybrid import assert_hybrid_attention_schedule
from longcontext.model.attention_utils import apply_block_rope, causal_compressed_block_mask
from longcontext.model.config import AttentionConfig, LongContextConfig, RopeConfig


def tiny_config(attention_type: str = "csa", scaling_factor: float = 1.0) -> LongContextConfig:
    return LongContextConfig(
        name=f"tiny_{attention_type}",
        vocab_size=64,
        n_layers=3,
        d_model=32,
        n_heads=2,
        d_head=16,
        d_ff=64,
        max_seq_len=16,
        rope=RopeConfig(scaling_factor=scaling_factor),
        attention=AttentionConfig(
            type=attention_type,
            local_window=4,
            top_k=2,
            index_dim=8,
            csa={"compression_block_size": 4, "top_k": 2, "local_window": 4, "index_dim": 8},
            hca={"compression_block_size": 64, "local_window": 4},
        ),
    )


def test_csa_topk_mask_uses_negative_infinity_before_selection():
    mask = causal_compressed_block_mask(seq_len=8, num_blocks=2, block_size=4, device=torch.device("cpu"))
    scores = torch.zeros(1, 8, 2).masked_fill(~mask[None], float("-inf"))
    topk_scores, _ = torch.topk(scores, k=1, dim=-1)

    assert mask[0].tolist() == [False, False]
    assert not torch.isfinite(topk_scores[0, 0, 0])
    assert mask[4].tolist() == [True, False]
    assert torch.isfinite(topk_scores[0, 4, 0])


def test_csa_beta_raw_initializes_to_logit_point_one():
    attn = CSALiteAttention(tiny_config("csa"))

    assert torch.allclose(attn.beta_raw.detach(), torch.tensor(math.log(0.1 / 0.9)))


def test_hca_compressed_block_rope_positions_are_block_ends(monkeypatch):
    captured: dict[str, torch.Tensor] = {}

    def fake_build_rope_freqs_for_positions(
        positions: torch.Tensor,
        rope_dim: int,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["positions"] = positions.detach().cpu()
        return torch.ones(positions.numel(), rope_dim // 2), torch.zeros(positions.numel(), rope_dim // 2)

    monkeypatch.setattr(
        "longcontext.model.attention_utils.build_rope_freqs_for_positions",
        fake_build_rope_freqs_for_positions,
    )
    k_comp = torch.randn(1, 3, 2, 16)

    apply_block_rope(k_comp, block_size=64, base=10000.0, scaling_factor=1.0)

    assert captured["positions"].tolist() == [63, 127, 191]


def test_hybrid_24_layer_schedule_counts_match_spec():
    counts = assert_hybrid_attention_schedule(24)

    assert counts == {"swa": 2, "hca": 7, "csa": 15}


def test_rope_scaling_factor_is_forwarded_from_config(monkeypatch):
    captured: dict[str, float] = {}

    def fake_build_rope_freqs(
        seq_len: int,
        rope_dim: int,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["scaling_factor"] = scaling_factor
        return torch.ones(seq_len, rope_dim // 2, device=device), torch.zeros(
            seq_len,
            rope_dim // 2,
            device=device,
        )

    monkeypatch.setattr("longcontext.model.attention_full.build_rope_freqs", fake_build_rope_freqs)
    attn = FullAttention(tiny_config("full", scaling_factor=4.0))

    attn(torch.randn(1, 4, 32))

    assert captured["scaling_factor"] == 4.0
