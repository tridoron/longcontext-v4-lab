from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_val_doc_hashes(path: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> set[str]:
    hashes = sorted({row["doc_hash"] for rows in rows_by_split.values() for row in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return set(hashes)


def split_validation(
    accepted_rows: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in accepted_rows:
        by_source.setdefault(row["source"], []).append(row)
    for rows in by_source.values():
        rows.sort(key=lambda item: item["doc_hash"])

    validation: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    validation_hashes: set[str] = set()
    split_stats: dict[str, Any] = {"validation": {}, "train_docs": 0, "train_tokens": 0}

    for val_name, source_quotas in quotas.items():
        split_stats["validation"][val_name] = {}
        for source, quota in source_quotas.items():
            total = 0
            selected: list[dict[str, Any]] = []
            for row in by_source.get(source, []):
                if row["doc_hash"] in validation_hashes:
                    continue
                selected.append(row)
                validation_hashes.add(row["doc_hash"])
                total += int(row.get("approx_tokens", 0))
                if total >= quota:
                    break
            if total < quota:
                raise RuntimeError(
                    f"{val_name}/{source} validation quota 未填满: got={total}, required={quota}"
                )
            validation[val_name].extend(selected)
            split_stats["validation"][val_name][source] = {
                "docs": len(selected),
                "tokens": total,
                "target_tokens": quota,
            }

    train_rows = [row for row in accepted_rows if row["doc_hash"] not in validation_hashes]
    split_stats["train_docs"] = len(train_rows)
    split_stats["train_tokens"] = sum(int(row.get("approx_tokens", 0)) for row in train_rows)
    split_stats["validation_doc_hashes"] = len(validation_hashes)
    return train_rows, validation, split_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="按 doc_hash 和固定 quota 构建 validation split")
    parser.add_argument("--input", default="data/metadata/accepted_samples.jsonl")
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--output-dir", default="data/metadata")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    quotas = {name: spec["mix"] for name, spec in cfg["validation"].items()}
    train_rows, validation, stats = split_validation(load_jsonl(Path(args.input)), quotas)
    out_dir = Path(args.output_dir)
    write_val_doc_hashes(out_dir / "val_doc_hashes.txt", validation)
    write_jsonl(out_dir / "train_docs.jsonl", train_rows)
    for name, rows in validation.items():
        write_jsonl(out_dir / f"{name}_docs.jsonl", rows)
    (out_dir / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
