from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="将 JSONL 文档 tokenized 为 uint32 token stream")
    parser.add_argument("--input", default="data/metadata/accepted_samples.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/shards")
    parser.add_argument("--prefix", default="train")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    doc_sep = tokenizer.token_to_id("<|doc_sep|>")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list[int]] = {}
    docs_by_source: dict[str, int] = defaultdict(int)
    tokens_by_source: dict[str, int] = defaultdict(int)
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = row["source"]
            ids = tokenizer.encode(row["text"]).ids[: int(row.get("max_tokens", 65536))]
            buffers.setdefault(source, []).extend(ids + [doc_sep])
            docs_by_source[source] += 1
            tokens_by_source[source] += len(ids)
    for source, ids in buffers.items():
        arr = np.asarray(ids, dtype=np.uint32)
        arr.tofile(out_dir / f"{args.prefix}_{source}_00000.bin")
    meta_dir = out_dir.parent / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    corpus_stats = {
        "prefix": args.prefix,
        "docs_by_source": dict(docs_by_source),
        "tokens_by_source": dict(tokens_by_source),
        "total_tokens": int(sum(tokens_by_source.values())),
    }
    (meta_dir / "corpus_stats.json").write_text(
        json.dumps(corpus_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = max(corpus_stats["total_tokens"], 1)
    source_mix = {source: tokens / total for source, tokens in tokens_by_source.items()}
    (meta_dir / "source_mix.json").write_text(
        json.dumps(source_mix, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
