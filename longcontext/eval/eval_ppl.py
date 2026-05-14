from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from data.dataset import PackedMemmapDataset
from longcontext.eval.common import load_config_and_model

IGNORE_INDEX = -100


EVAL_FILES = {
    1024: "data/shards/val_pretrain_seq1024.bin",
    4096: "data/shards/val_pretrain_seq4096.bin",
    8192: "data/shards/val_long_seq8192.bin",
    16384: "data/shards/val_long_seq16384.bin",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def resolve_eval_file(seq_len: int, data_file: str | None, sources_config: str) -> Path:
    if data_file:
        return resolve_project_path(data_file)
    sources_path = resolve_project_path(sources_config)
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    key = f"val_pretrain_seq{seq_len}" if seq_len <= 4096 else f"val_long_seq{seq_len}"
    shard_spec = sources.get("validation_shards", {}).get(key)
    if shard_spec is None:
        return resolve_project_path(EVAL_FILES[seq_len])
    shard_path = shard_spec["path"] if isinstance(shard_spec, dict) else shard_spec
    return resolve_project_path(shard_path)


@torch.no_grad()
def evaluate_ppl(
    config_path: str,
    weights_path: str | None,
    seq_len: int,
    data_file: str | None = None,
    batch_size: int = 1,
    max_batches: int | None = None,
    sources_config: str = "configs/data_sources.yaml",
) -> dict:
    config, model, device = load_config_and_model(config_path, weights_path)
    path = resolve_eval_file(seq_len, data_file, sources_config)
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
        tokens = labels.ne(IGNORE_INDEX).sum().item()
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
    parser.add_argument("--sources", default="configs/data_sources.yaml")
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
        args.sources,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
