from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def load_val_doc_hashes(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"训练 shard 构建前必须先生成验证集 doc hash 文件: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def assert_not_validation_doc(row: dict, val_hashes: set[str], line_no: int) -> None:
    doc_hash = row.get("doc_hash")
    if not doc_hash:
        raise RuntimeError(f"第 {line_no} 行缺少 doc_hash，无法检查 train/validation 泄漏")
    if doc_hash in val_hashes:
        raise RuntimeError(f"训练 shard 包含验证集文档: line={line_no}, doc_hash={doc_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 JSONL 文档 tokenized 为 uint32 token stream")
    parser.add_argument("--input", default="data/metadata/accepted_samples.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/shards")
    parser.add_argument("--prefix", default="train")
    parser.add_argument("--val-doc-hashes", default="data/metadata/val_doc_hashes.txt")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    doc_sep = tokenizer.token_to_id("<|doc_sep|>")
    if doc_sep is None:
        raise RuntimeError("tokenizer 缺少 <|doc_sep|> token")
    val_hashes = load_val_doc_hashes(Path(args.val_doc_hashes)) if args.prefix == "train" else set()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list[int]] = {}
    docs_by_source: dict[str, int] = defaultdict(int)
    tokens_by_source: dict[str, int] = defaultdict(int)
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if args.prefix == "train":
                assert_not_validation_doc(row, val_hashes, line_no)
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
