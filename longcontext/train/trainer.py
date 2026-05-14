from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from longcontext.model.transformer import LongContextLM
from longcontext.train.checkpoint import save_model_weights, save_training_state, strip_optimizer_state

IGNORE_INDEX = -100


class Trainer:
    def __init__(
        self,
        model: LongContextLM,
        optimizer: Any,
        train_loader: Iterable,
        device: torch.device,
        output_dir: str | Path,
        scheduler: Any | None = None,
        grad_clip: float = 1.0,
        amp_dtype: str = "bf16",
        grad_accum_steps: int = 1,
        milestone_tokens: Iterable[int] | None = None,
        target_seen_tokens: int | None = None,
        strip_optimizer_state_on_finish: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        if device.type == "cuda" and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            device_name = torch.cuda.get_device_name(device)
            raise RuntimeError(f"当前 CUDA 设备不支持 bf16 autocast: {device_name}")
        self.milestone_tokens = sorted(
            {int(token) for token in milestone_tokens or [] if int(token) > 0}
        )
        self.target_seen_tokens = int(target_seen_tokens) if target_seen_tokens is not None else None
        self.strip_optimizer_state_on_finish = strip_optimizer_state_on_finish
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _valid_tokens(labels: torch.Tensor) -> int:
        return int(labels.ne(IGNORE_INDEX).sum().item())

    @staticmethod
    def _milestone_suffix(tokens: int) -> str:
        if tokens % 1_000_000_000 == 0:
            return f"{tokens // 1_000_000_000}b"
        if tokens % 1_000_000 == 0:
            return f"{tokens // 1_000_000}m"
        if tokens % 1_000 == 0:
            return f"{tokens // 1_000}k"
        return str(tokens)

    def train(self, max_steps: int, save_every: int = 1000) -> dict[str, Any]:
        self.model.train()
        log_path = self.output_dir / "train_log.jsonl.gz"
        seen_tokens = 0
        start = time.perf_counter()
        last_loss = 0.0
        with gzip.open(log_path, "at", encoding="utf-8") as log_f:
            iterator = iter(self.train_loader)
            pending_milestones = list(self.milestone_tokens)
            last_step = 0
            for step in range(1, max_steps + 1):
                last_step = step
                micro_batches = []
                total_loss_tokens = 0
                for _ in range(self.grad_accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.train_loader)
                        batch = next(iterator)
                    valid_tokens = self._valid_tokens(batch["labels"])
                    micro_batches.append((batch, valid_tokens))
                    total_loss_tokens += valid_tokens
                if total_loss_tokens <= 0:
                    raise RuntimeError("当前梯度累积窗口没有任何有效 token，无法归一化 loss")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss_sum = 0.0
                for batch, _valid_tokens in micro_batches:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.device.type == "cuda",
                    ):
                        out = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                    assert out.loss_sum is not None
                    (out.loss_sum / total_loss_tokens).backward()
                    total_loss_sum += float(out.loss_sum.detach().cpu())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                seen_tokens += total_loss_tokens
                last_loss = total_loss_sum / total_loss_tokens
                elapsed = max(time.perf_counter() - start, 1e-9)
                row = {
                    "step": step,
                    "loss": last_loss,
                    "grad_norm": float(grad_norm),
                    "seen_tokens": seen_tokens,
                    "tokens_per_second": seen_tokens / elapsed,
                }
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                while pending_milestones and seen_tokens >= pending_milestones[0]:
                    milestone = pending_milestones.pop(0)
                    suffix = self._milestone_suffix(milestone)
                    save_model_weights(self.model, self.output_dir / f"weights_{suffix}.safetensors")
                if save_every and step % save_every == 0:
                    save_training_state(
                        self.output_dir / "state_latest.pt",
                        self.model,
                        self.optimizer,
                        step,
                        seen_tokens,
                        scheduler=self.scheduler,
                    )
                if self.target_seen_tokens is not None and seen_tokens >= self.target_seen_tokens:
                    break
        save_model_weights(self.model, self.output_dir / "weights_final.safetensors")
        state_path = self.output_dir / "state_latest.pt"
        save_training_state(
            state_path,
            self.model,
            self.optimizer,
            last_step,
            seen_tokens,
            scheduler=self.scheduler,
        )
        if self.strip_optimizer_state_on_finish:
            strip_optimizer_state(state_path)
        return {"loss": last_loss, "seen_tokens": seen_tokens}
