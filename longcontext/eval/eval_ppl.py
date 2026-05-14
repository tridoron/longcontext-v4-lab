from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data.dataset import PackedMemmapDataset
from longcontext.eval.common import load_config_and_model


EVAL_FILES = {
    1024: "data/shards/val_pretrain_seq1024.bin",
    4096: "data/shards/val_pretrain_seq4096.bin",
    8192: "data/shards/val_long_seq8192.bin",
    16384: "data/shards/val_long_seq16384.bin",
}


@torch.no_grad()
def evaluate_ppl(
    config_path: str,
    weights_path: str | None,
    seq_len: int,
    data_file: str | None = None,
    batch_size: int = 1,
    max_batches: int | None = None,
) -> dict:
    config, model, device = load_config_and_model(config_path, weights_path)
    path = data_file or EVAL_FILES[seq_len]
    dataset = PackedMemmapDataset(path, seq_len=seq_len, pad_token_id=config.pad_token_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids, labels=labels, attention_mask=attention_mask)
        assert out.loss is not None
        tokens = labels.ne(config.pad_token_id).sum().item()
        total_loss += float(out.loss) * tokens
        total_tokens += tokens
    loss = total_loss / max(total_tokens, 1)
    return {"model": config.name, "seq_len": seq_len, "loss": loss, "ppl": math.exp(loss), "num_tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description="PPL 评测")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--seq-len", type=int, required=True, choices=sorted(EVAL_FILES))
    parser.add_argument("--data-file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", default="outputs/artifacts/ppl_result.json")
    args = parser.parse_args()
    result = evaluate_ppl(
        args.config,
        args.weights,
        args.seq_len,
        args.data_file,
        args.batch_size,
        args.max_batches,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
