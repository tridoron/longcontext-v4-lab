from __future__ import annotations

import torch

from longcontext.model.config import AttentionConfig, LongContextConfig, MHCConfig
from longcontext.model.transformer import LongContextLM
from longcontext.optim.build_optimizer import build_optimizer, split_parameter_groups


def tiny_config(attention_type: str = "full", mhc: bool = False) -> LongContextConfig:
    return LongContextConfig(
        name=f"tiny_{attention_type}",
        vocab_size=128,
        n_layers=3,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        attention=AttentionConfig(
            type=attention_type,
            local_window=8,
            top_k=4,
            index_dim=16,
            csa={"compression_block_size": 4, "top_k": 4, "local_window": 8, "index_dim": 16},
            hca={"compression_block_size": 8, "local_window": 8},
        ),
        mhc=MHCConfig(enabled=mhc, n_hc=2, sinkhorn_iters=3),
    )


def test_attention_variants_forward_backward_step(tmp_path):
    for attention_type in ["full", "swa", "csa", "hca", "hybrid"]:
        config = tiny_config(attention_type)
        model = LongContextLM(config)
        opt = build_optimizer(model, {"lr": 1e-4, "use_muon": True}, log_dir=tmp_path / attention_type)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        out = model(input_ids, labels=labels)
        assert out.logits.shape == (2, 16, config.vocab_size)
        assert out.loss is not None
        out.loss.backward()
        opt.step()


def test_mhc_forward():
    config = tiny_config("hybrid", mhc=True)
    model = LongContextLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    out = model(input_ids)
    assert out.logits.shape == (1, 16, config.vocab_size)


def test_beta_raw_not_in_muon():
    config = tiny_config("csa")
    model = LongContextLM(config)
    _, _, _, muon_names, adamw_names = split_parameter_groups(model, use_muon=True)
    assert all("beta_raw" not in name for name in muon_names)
    assert any("beta_raw" in name for name in adamw_names)


def test_loss_ignores_negative_100_labels():
    config = tiny_config("full")
    model = LongContextLM(config)
    input_ids = torch.randint(1, config.vocab_size, (1, 4))
    labels = torch.tensor([[-100, 2, 3, -100]])

    out = model(input_ids, labels=labels)
    expected = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    assert out.loss is not None
    assert out.loss_sum is not None
    assert out.loss_tokens is not None
    assert torch.allclose(out.loss, expected)
    assert out.loss_tokens.item() == 2


def test_gradient_checkpointing_toggle():
    config = tiny_config("full")
    model = LongContextLM(config)

    model.gradient_checkpointing_enable()
    assert model.config.gradient_checkpointing

    model.gradient_checkpointing_disable()
    assert not model.config.gradient_checkpointing
