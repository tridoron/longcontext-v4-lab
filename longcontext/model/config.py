from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RopeConfig:
    base: float = 10000.0
    scaling_type: str = "linear_position"
    scaling_factor: float = 1.0


@dataclass
class AttentionConfig:
    type: str = "full"
    local_window: int = 256
    compression_block_size: int = 4
    top_k: int = 128
    index_dim: int = 128
    swa_layers: int = 2
    csa: dict[str, Any] = field(default_factory=dict)
    hca: dict[str, Any] = field(default_factory=dict)


@dataclass
class MHCConfig:
    enabled: bool = False
    n_hc: int = 2
    sinkhorn_iters: int = 10


@dataclass
class LongContextConfig:
    name: str = "debug_120m"
    vocab_size: int = 32000
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope: RopeConfig = field(default_factory=RopeConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mhc: MHCConfig = field(default_factory=MHCConfig)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False
    tie_embeddings: bool = False

    @property
    def attention_type(self) -> str:
        return self.attention.type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LongContextConfig":
        model = raw.get("model", raw)
        rope_raw = raw.get("rope", model.get("rope", {}))
        attn_raw = raw.get("attention", model.get("attention", {}))
        mhc_raw = raw.get("mhc", model.get("mhc", {}))
        if "attention_type" in model and "type" not in attn_raw:
            attn_raw = {**attn_raw, "type": model["attention_type"]}
        if "rope_base" in model and "base" not in rope_raw:
            rope_raw = {**rope_raw, "base": model["rope_base"]}
        return cls(
            name=model.get("name", "debug_120m"),
            vocab_size=int(model.get("vocab_size", 32000)),
            n_layers=int(model.get("n_layers", 12)),
            d_model=int(model.get("d_model", 768)),
            n_heads=int(model.get("n_heads", 12)),
            d_head=int(model.get("d_head", 64)),
            d_ff=int(model.get("d_ff", 2048)),
            max_seq_len=int(model.get("max_seq_len", 1024)),
            rope=RopeConfig(**rope_raw),
            attention=AttentionConfig(**attn_raw),
            mhc=MHCConfig(**mhc_raw),
            pad_token_id=int(model.get("pad_token_id", 0)),
            bos_token_id=int(model.get("bos_token_id", 1)),
            eos_token_id=int(model.get("eos_token_id", 2)),
            gradient_checkpointing=bool(raw.get("training", {}).get("gradient_checkpointing", False)),
            tie_embeddings=bool(model.get("tie_embeddings", False)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LongContextConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.name,
                "vocab_size": self.vocab_size,
                "n_layers": self.n_layers,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "d_head": self.d_head,
                "d_ff": self.d_ff,
                "max_seq_len": self.max_seq_len,
                "attention_type": self.attention.type,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "tie_embeddings": self.tie_embeddings,
            },
            "rope": self.rope.__dict__,
            "attention": self.attention.__dict__,
            "mhc": self.mhc.__dict__,
        }
