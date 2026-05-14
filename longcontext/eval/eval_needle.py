from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from longcontext.eval.common import load_config_and_model, write_csv


def build_needle_text(target_tokens: int, position_ratio: float) -> str:
    filler = "This haystack sentence is unrelated context for a long sequence benchmark. "
    needle = "John's favorite programming language is Rust."
    question = "What is John's favorite programming language?"
    parts = [filler] * max(target_tokens, 128)
    idx = min(len(parts) - 1, int(len(parts) * position_ratio))
    parts.insert(idx, needle)
    parts.append(question)
    return " ".join(parts)


@torch.no_grad()
def run_eval(config_path: str, weights_path: str | None, tokenizer_path: str, lengths: list[int], samples: int) -> list[dict]:
    config, model, device = load_config_and_model(config_path, weights_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    rows = []
    for length in lengths:
        for pos in [0.1, 0.3, 0.5, 0.7, 0.9]:
            correct = 0
            for _ in range(samples):
                ids = tokenizer.encode(build_needle_text(length, pos)).ids[: config.max_seq_len]
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                decoded = tokenizer.decode(model.generate(input_ids, max_new_tokens=16)[0].tolist())
                correct += int("Rust" in decoded or "rust" in decoded)
            rows.append(
                {
                    "model": config.name,
                    "length": length,
                    "position": pos,
                    "accuracy": correct / max(samples, 1),
                    "samples": samples,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack 评测")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", default="outputs/artifacts/needle_results.csv")
    args = parser.parse_args()
    write_csv(args.output, run_eval(args.config, args.weights, args.tokenizer, args.lengths, args.samples))


if __name__ == "__main__":
    main()
