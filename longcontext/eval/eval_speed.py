from __future__ import annotations

import argparse
import time

import torch

from longcontext.eval.common import load_config_and_model, write_csv
from longcontext.utils.profiler import cuda_memory_stats


@torch.no_grad()
def benchmark(config_path: str, weights_path: str | None, lengths: list[int], steps: int) -> list[dict]:
    config, model, device = load_config_and_model(config_path, weights_path)
    rows = []
    for seq_len in lengths:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            _ = model(input_ids).logits
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(input_ids).logits
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-9)
        mem = cuda_memory_stats()
        rows.append(
            {
                "model": config.name,
                "seq_len": seq_len,
                "prefill_tokens_per_second": seq_len * steps / elapsed,
                "decode_tokens_per_second": steps / elapsed,
                "allocated_cuda_memory_mb": mem.allocated_mb,
                "peak_memory_mb": mem.peak_mb,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="速度与显存 smoke benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="outputs/artifacts/speed_results.csv")
    args = parser.parse_args()
    rows = benchmark(args.config, args.weights, args.lengths, args.steps)
    write_csv(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
