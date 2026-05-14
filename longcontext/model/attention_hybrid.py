from __future__ import annotations

import logging

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
