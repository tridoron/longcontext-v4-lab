from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class MemoryStats:
    allocated_mb: float
    peak_mb: float


def cuda_memory_stats() -> MemoryStats:
    if not torch.cuda.is_available():
        return MemoryStats(allocated_mb=0.0, peak_mb=0.0)
    return MemoryStats(
        allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024,
        peak_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
    )


@contextmanager
def timed():
    start = time.perf_counter()
    box = {"elapsed": 0.0}
    try:
        yield box
    finally:
        box["elapsed"] = time.perf_counter() - start
