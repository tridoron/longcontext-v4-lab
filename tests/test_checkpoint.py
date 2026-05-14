from __future__ import annotations

import torch

from longcontext.train.checkpoint import load_training_state, save_training_state, strip_optimizer_state


def test_training_state_restores_scheduler_and_rng(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)

    optimizer.step()
    scheduler.step()
    torch.manual_seed(1234)
    expected_next = torch.rand(4)

    torch.manual_seed(1234)
    path = tmp_path / "state.pt"
    save_training_state(path, model, optimizer, step=7, seen_tokens=128, scheduler=scheduler)
    torch.rand(4)

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=1.0)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda step: 0.5 ** step,
    )

    state = load_training_state(path, restored_model, restored_optimizer, restored_scheduler)

    assert state["step"] == 7
    assert state["seen_tokens"] == 128
    assert state["scheduler"] is not None
    assert "cuda_rng_state" in state
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert torch.allclose(torch.rand(4), expected_next)


def test_strip_optimizer_state_keeps_model_and_seen_tokens(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    path = tmp_path / "state_latest.pt"

    save_training_state(path, model, optimizer, step=3, seen_tokens=256)
    strip_optimizer_state(path)

    state = torch.load(path, map_location="cpu")
    assert state["model"] is not None
    assert state["optimizer"] is None
    assert state["scheduler"] is None
    assert state["step"] == 3
    assert state["seen_tokens"] == 256
