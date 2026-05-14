from __future__ import annotations

from dataclasses import dataclass

from longcontext.model.attention_hybrid import hybrid_attention_type
from longcontext.model.config import LongContextConfig


@dataclass
class KVCacheEstimate:
    seq_len: int
    bytes_per_token: int
    dense_kv_mb: float
    estimated_kv_mb: float
    compression_ratio: float
    attention_compressed_length: int
    csa_selected_block_count: int
    hca_compressed_block_count: int


def layer_cache_tokens(config: LongContextConfig, seq_len: int, attention_type: str) -> int:
    if attention_type == "full":
        return seq_len
    if attention_type == "swa":
        return min(seq_len, config.attention.local_window)
    if attention_type == "csa":
        block = int(config.attention.csa.get("compression_block_size", 4))
        top_k = int(config.attention.csa.get("top_k", config.attention.top_k))
        local = int(config.attention.csa.get("local_window", config.attention.local_window))
        return min(seq_len, local) + min(top_k, (seq_len + block - 1) // block)
    if attention_type == "hca":
        block = int(config.attention.hca.get("compression_block_size", 64))
        local = int(config.attention.hca.get("local_window", config.attention.local_window))
        return min(seq_len, local) + (seq_len + block - 1) // block
    raise ValueError(f"未知 attention 类型: {attention_type}")


def estimate_kv_cache(
    config: LongContextConfig,
    seq_len: int,
    dtype_bytes: int = 2,
) -> KVCacheEstimate:
    bytes_per_entry = 2 * config.n_heads * config.d_head * dtype_bytes
    dense_tokens = config.n_layers * seq_len
    estimated_tokens = 0
    csa_selected = 0
    hca_blocks = 0
    for layer_id in range(config.n_layers):
        attn_type = config.attention.type
        if attn_type == "hybrid":
            attn_type = hybrid_attention_type(layer_id)
        estimated_tokens += layer_cache_tokens(config, seq_len, attn_type)
        if attn_type == "csa":
            block = int(config.attention.csa.get("compression_block_size", 4))
            csa_selected = max(csa_selected, min(config.attention.top_k, (seq_len + block - 1) // block))
        if attn_type == "hca":
            block = int(config.attention.hca.get("compression_block_size", 64))
            hca_blocks = max(hca_blocks, (seq_len + block - 1) // block)
    dense_mb = dense_tokens * bytes_per_entry / 1024 / 1024
    est_mb = estimated_tokens * bytes_per_entry / 1024 / 1024
    return KVCacheEstimate(
        seq_len=seq_len,
        bytes_per_token=bytes_per_entry,
        dense_kv_mb=dense_mb,
        estimated_kv_mb=est_mb,
        compression_ratio=dense_mb / max(est_mb, 1e-9),
        attention_compressed_length=estimated_tokens // max(config.n_layers, 1),
        csa_selected_block_count=csa_selected,
        hca_compressed_block_count=hca_blocks,
    )
