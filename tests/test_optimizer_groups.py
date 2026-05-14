from __future__ import annotations

import math

import pytest
import torch

from longcontext.model.config import AttentionConfig, LongContextConfig
from longcontext.model.transformer import LongContextLM
from longcontext.optim.build_optimizer import build_optimizer, collect_parameter_groups
from longcontext.optim.muon import Muon, zeropower_via_newton_schulz_5


def tiny_csa_config() -> LongContextConfig:
    return LongContextConfig(
        name="tiny_csa_optimizer",
        vocab_size=64,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_head=16,
        d_ff=64,
        max_seq_len=16,
        attention=AttentionConfig(
            type="csa",
            local_window=4,
            top_k=2,
            index_dim=8,
            csa={"compression_block_size": 4, "top_k": 2, "local_window": 4, "index_dim": 8},
        ),
    )


def test_optimizer_groups_follow_muon_contract():
    model = LongContextLM(tiny_csa_config())
    groups = collect_parameter_groups(model, use_muon=True)

    assert groups.beta_no_decay
    assert all("beta_raw" not in name for name in groups.muon_names)
    assert any(name.endswith("attn.w_z.weight") for name in groups.muon_names)
    assert all("tok_embeddings" not in name for name in groups.muon_names)
    assert all("lm_head" not in name for name in groups.muon_names)
    assert all(param.ndim == 2 for param in groups.muon_params)


def test_beta_raw_has_dedicated_adamw_no_decay_group():
    model = LongContextLM(tiny_csa_config())
    optimizer = build_optimizer(model, {"lr": 1e-4, "use_muon": True})
    adamw = next(opt for opt in optimizer.optimizers if isinstance(opt, torch.optim.AdamW))
    beta_groups = [group for group in adamw.param_groups if group.get("name") == "beta_raw"]

    assert len(beta_groups) == 1
    assert beta_groups[0]["weight_decay"] == 0.0
    assert len(beta_groups[0]["params"]) == 2
    assert all(param.ndim == 0 for param in beta_groups[0]["params"])


def test_muon_nesterov_update_and_ns_steps_are_configurable(monkeypatch):
    captured: dict[str, int] = {}

    def fake_orthogonalize(update: torch.Tensor, eps: float = 1e-7, ns_steps: int = 5) -> torch.Tensor:
        captured["ns_steps"] = ns_steps
        return update

    monkeypatch.setattr("longcontext.optim.muon.zeropower_via_newton_schulz_5", fake_orthogonalize)
    param = torch.nn.Parameter(torch.zeros(2, 2))
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    param.grad = grad.clone()
    optimizer = Muon(
        [param],
        lr=1.0,
        weight_decay=0.0,
        momentum=0.9,
        nesterov=True,
        ns_steps=3,
        update_scale=1.0 / math.sqrt(2.0),
    )

    optimizer.step()

    assert captured["ns_steps"] == 3
    assert torch.allclose(param, -1.9 * grad)


def test_muon_rejects_non_2d_params_and_invalid_ns_steps():
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.ones_like(param)
    optimizer = Muon([param])

    with pytest.raises(ValueError, match="二维矩阵"):
        optimizer.step()
    with pytest.raises(ValueError, match="ns_steps"):
        zeropower_via_newton_schulz_5(torch.randn(2, 2), ns_steps=0)
