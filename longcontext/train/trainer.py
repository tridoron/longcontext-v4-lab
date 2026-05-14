from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from longcontext.model.transformer import LongContextLM
from longcontext.train.checkpoint import save_model_weights, save_training_state


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
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.grad_clip = grad_clip
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, max_steps: int, save_every: int = 1000) -> dict[str, Any]:
        self.model.train()
        log_path = self.output_dir / "train_log.jsonl.gz"
        seen_tokens = 0
        start = time.perf_counter()
        last_loss = 0.0
        with gzip.open(log_path, "at", encoding="utf-8") as log_f:
            iterator = iter(self.train_loader)
            for step in range(1, max_steps + 1):
                batch = next(iterator)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
                    out = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                assert out.loss is not None
                out.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                tokens = input_ids.numel()
                seen_tokens += tokens
                last_loss = float(out.loss.detach().cpu())
                elapsed = max(time.perf_counter() - start, 1e-9)
                row = {
                    "step": step,
                    "loss": last_loss,
                    "grad_norm": float(grad_norm),
                    "seen_tokens": seen_tokens,
                    "tokens_per_second": seen_tokens / elapsed,
                }
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if save_every and step % save_every == 0:
                    save_training_state(
                        self.output_dir / "state_latest.pt",
                        self.model,
                        self.optimizer,
                        step,
                        seen_tokens,
                        scheduler=self.scheduler,
                    )
        save_model_weights(self.model, self.output_dir / "weights_final.safetensors")
        return {"loss": last_loss, "seen_tokens": seen_tokens}
