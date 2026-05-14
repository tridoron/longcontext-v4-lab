from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from longcontext.model.block import TransformerBlock
from longcontext.model.config import LongContextConfig
from longcontext.model.mhc import FinalMHCProjector
from longcontext.model.rmsnorm import RMSNorm


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class LongContextLM(nn.Module):
    def __init__(self, config: LongContextConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_mhc = FinalMHCProjector(config.mhc.n_hc) if config.mhc.enabled else None
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def gradient_checkpointing_enable(self) -> None:
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.config.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LMOutput:
        x = self.tok_embeddings(input_ids)
        if self.config.mhc.enabled:
            x = x.unsqueeze(2).expand(-1, -1, self.config.mhc.n_hc, -1).contiguous()
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        if self.final_mhc is not None:
            x = self.final_mhc(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return LMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.config.max_seq_len :]
            logits = self(ctx).logits[:, -1]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat((out, next_token), dim=-1)
            if torch.all(next_token == eos):
                break
        return out
