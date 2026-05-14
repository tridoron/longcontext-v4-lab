from __future__ import annotations

import torch

from longcontext.model.config import AttentionConfig, LongContextConfig
from longcontext.model.transformer import LongContextLM
from longcontext.train.trainer import Trainer


def tiny_config() -> LongContextConfig:
    return LongContextConfig(
        name="tiny_trainer",
        vocab_size=32,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_head=8,
        d_ff=32,
        max_seq_len=4,
        attention=AttentionConfig(type="full", local_window=4),
    )


def test_trainer_seen_tokens_uses_effective_label_tokens(tmp_path):
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 0, 0]]),
            "labels": torch.tensor([[2, -100, -100, -100]]),
            "attention_mask": torch.tensor([[True, True, False, False]]),
        },
        {
            "input_ids": torch.tensor([[3, 4, 5, 0]]),
            "labels": torch.tensor([[4, 5, -100, -100]]),
            "attention_mask": torch.tensor([[True, True, True, False]]),
        },
    ]
    model = LongContextLM(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model,
        optimizer,
        batches,
        torch.device("cpu"),
        tmp_path,
        grad_accum_steps=2,
        milestone_tokens=[3],
    )

    result = trainer.train(max_steps=1, save_every=0)

    assert result["seen_tokens"] == 3
    assert (tmp_path / "weights_3.safetensors").exists()
    state = torch.load(tmp_path / "state_latest.pt", map_location="cpu")
    assert state["seen_tokens"] == 3
