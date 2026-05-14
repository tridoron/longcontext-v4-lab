from __future__ import annotations

import logging
from collections import Counter

from torch import nn

from longcontext.model.attention_csa import CSALiteAttention
from longcontext.model.attention_hca import HCALiteAttention
from longcontext.model.attention_swa import SlidingWindowAttention
from longcontext.model.config import LongContextConfig

LOGGER = logging.getLogger(__name__)


def hybrid_attention_type(layer_id: int) -> str:
    if layer_id < 2:
        return "swa"
    if layer_id % 3 == 0:
        return "hca"
    return "csa"


def hybrid_attention_schedule(n_layers: int) -> list[str]:
    return [hybrid_attention_type(layer_id) for layer_id in range(n_layers)]


def assert_hybrid_attention_schedule(n_layers: int) -> dict[str, int]:
    schedule = hybrid_attention_schedule(n_layers)
    if n_layers >= 2 and schedule[:2] != ["swa", "swa"]:
        raise AssertionError("Hybrid attention 调度错误: layer 0,1 必须为 SWA")
    for layer_id, attention_type in enumerate(schedule):
        expected = "swa" if layer_id < 2 else "hca" if layer_id % 3 == 0 else "csa"
        if attention_type != expected:
            raise AssertionError(
                f"Hybrid attention 调度错误: layer={layer_id}, got={attention_type}, expected={expected}"
            )
    counts = dict(Counter(schedule))
    if n_layers == 24:
        expected_counts = {"swa": 2, "hca": 7, "csa": 15}
        if counts != expected_counts:
            raise AssertionError(f"Hybrid attention 24 层分布错误: got={counts}, expected={expected_counts}")
    LOGGER.info("hybrid attention schedule n_layers=%d counts=%s", n_layers, counts)
    return counts


class HybridAttention(nn.Module):
    impl: nn.Module

    def __init__(self, config: LongContextConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention_type = hybrid_attention_type(layer_id)
        if self.attention_type == "swa":
            self.impl = SlidingWindowAttention(config, layer_id)
        elif self.attention_type == "hca":
            self.impl = HCALiteAttention(config, layer_id)
        else:
            self.impl = CSALiteAttention(config, layer_id)
        LOGGER.info("layer %02d attention=%s", layer_id, self.attention_type.upper())

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)
