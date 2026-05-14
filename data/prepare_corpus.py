from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

REASONS = [
    "empty_text",
    "too_short_chars",
    "too_short_tokens",
    "too_few_lines",
    "duplicate_doc",
    "unsupported_language",
    "missing_code_metadata",
    "invalid_text",
]

SOURCE_RULES = {
    "web": {"min_chars": 300, "min_tokens": 128, "min_lines": 3, "max_tokens": 8192},
    "code": {"min_chars": 200, "min_tokens": 128, "min_lines": 5, "max_tokens": 8192},
    "math": {"min_chars": 300, "min_tokens": 128, "min_lines": 3, "max_tokens": 8192},
    "long_doc": {"min_chars": 2000, "min_tokens": 1024, "min_lines": 20, "max_tokens": 65536},
}

LANGUAGE_WHITELIST = [
    "python",
    "java",
    "javascript",
    "typescript",
    "cpp",
    "c",
    "go",
    "rust",
    "sql",
    "shell",
    "kotlin",
]

EXT_TO_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".go": "go",
    ".rs": "rust",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".kt": "kotlin",
    ".kts": "kotlin",
}

TARGET_CODE_TOKENS = {
    "python": 55_000_000,
    "java": 45_000_000,
    "javascript": 25_000_000,
    "typescript": 20_000_000,
    "cpp": 25_000_000,
    "c": 15_000_000,
    "go": 17_500_000,
    "rust": 12_500_000,
    "sql": 15_000_000,
    "shell": 12_500_000,
    "kotlin": 7_500_000,
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def doc_hash(source: str, normalized_text: str) -> str:
    return hashlib.sha1(f"{source}\n{normalized_text}".encode("utf-8")).hexdigest()


def infer_language(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    language = sample.get("language") or sample.get("lang")
    if language:
        language = str(language).lower()
        return language, None if language in LANGUAGE_WHITELIST else "unsupported_language"
    path = (
        sample.get("path")
        or sample.get("file_path")
        or sample.get("filename")
        or sample.get("max_stars_repo_path")
        or ""
    )
    suffix = Path(str(path)).suffix.lower()
    if not suffix:
        return None, "missing_code_metadata"
    language = EXT_TO_LANGUAGE.get(suffix)
    if language is None:
        return None, "unsupported_language"
    return language, None


def code_extra_reject_reason(text: str) -> str | None:
    if "\ufffd" in text:
        return "invalid_text"
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t")
    if controls / max(len(text), 1) > 0.02:
        return "invalid_text"
    lines = text.splitlines()
    if lines and sum(len(line) for line in lines) / len(lines) > 300:
        return "invalid_text"
    return None


def text_from_sample(sample: dict[str, Any]) -> str:
    for key in ("text", "content", "code", "document"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


def build_empty_filter_stats() -> dict[str, Any]:
    return {
        "accepted_docs": 0,
        "rejected_docs": 0,
        "rejected_by_reason": {reason: 0 for reason in REASONS},
        "accepted_tokens_by_source": {source: 0 for source in SOURCE_RULES},
    }


def validate_sample(
    sample: dict[str, Any],
    source: str,
    seen_hashes: set[str],
    tokenizer=None,
) -> tuple[dict[str, Any] | None, str | None]:
    raw_text = text_from_sample(sample)
    text = normalize_text(raw_text)
    if not text:
        return None, "empty_text"
    h = doc_hash(source, text)
    if h in seen_hashes:
        return None, "duplicate_doc"
    if source == "code":
        language, reason = infer_language(sample)
        if reason is not None:
            return None, reason
        reason = code_extra_reject_reason(text)
        if reason is not None:
            return None, reason
    else:
        language = None
    rule = SOURCE_RULES[source]
    if len(text) < rule["min_chars"]:
        return None, "too_short_chars"
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) < rule["min_lines"]:
        return None, "too_few_lines"
    token_count = len(tokenizer.encode(text).ids) if tokenizer is not None else len(text.split())
    if token_count < rule["min_tokens"]:
        return None, "too_short_tokens"
    seen_hashes.add(h)
    return {
        "source": source,
        "language": language,
        "doc_hash": h,
        "text": text,
        "approx_tokens": token_count,
        "max_tokens": rule["max_tokens"],
    }, None


def stream_regular_source(spec: dict[str, Any]):
    return load_dataset(
        spec["path"],
        spec.get("name"),
        split=spec.get("split", "train"),
        streaming=True,
    )


def stream_starcoder_source(spec: dict[str, Any]):
    repo = spec["path"]
    split = spec.get("split", "train")
    for language in spec.get("languages", LANGUAGE_WHITELIST):
        if language not in LANGUAGE_WHITELIST:
            raise ValueError(f"StarCoderData 语言不在白名单中: {language}")
        data_files = f"hf://datasets/{repo}/{language}/{split}-*.parquet"
        for sample in load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            streaming=True,
        ):
            row = dict(sample)
            row["language"] = language
            yield row


def stream_proof_pile_source(spec: dict[str, Any]):
    repo = spec["path"]
    subset = spec.get("name")
    split = spec.get("split", "train")
    if subset not in {"open-web-math", "arxiv"}:
        raise ValueError(f"不支持的 Proof-Pile-2 子集: {subset}")
    data_files = f"hf://datasets/{repo}/{subset}/{split}/*.jsonl.zst"
    return load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=True,
    )


def stream_source(source: str, spec: dict[str, Any]):
    if source == "code" and spec["path"] == "bigcode/starcoderdata" and "languages" in spec:
        return stream_starcoder_source(spec)
    if spec["path"] == "EleutherAI/proof-pile-2":
        return stream_proof_pile_source(spec)
    return stream_regular_source(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="流式清洗公开语料")
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--output", default="data/metadata/accepted_samples.jsonl")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = output.parent / "rejected_samples.jsonl"
    filter_stats = build_empty_filter_stats()
    code_stats: dict[str, Any] = {
        "target_tokens": TARGET_CODE_TOKENS,
        "actual_tokens": {lang: 0 for lang in LANGUAGE_WHITELIST},
        "accepted_docs": {},
        "rejected_docs": {},
        "language_whitelist": LANGUAGE_WHITELIST,
    }
    seen_hashes: set[str] = set()
    accepted = 0
    reached_limit = False
    with output.open("w", encoding="utf-8") as out_f, rejected_path.open("w", encoding="utf-8") as rej_f:
        for source, spec in cfg["sources"].items():
            for sample in stream_source(source, spec):
                row, reason = validate_sample(sample, source, seen_hashes)
                if row is None:
                    filter_stats["rejected_docs"] += 1
                    filter_stats["rejected_by_reason"][reason] += 1
                    if source == "code":
                        code_stats["rejected_docs"][reason] = code_stats["rejected_docs"].get(reason, 0) + 1
                    rej_f.write(json.dumps({"source": source, "reason": reason}, ensure_ascii=False) + "\n")
                    continue
                filter_stats["accepted_docs"] += 1
                filter_stats["accepted_tokens_by_source"][source] += row["approx_tokens"]
                if source == "code" and row["language"]:
                    lang = row["language"]
                    code_stats["actual_tokens"][lang] += row["approx_tokens"]
                    code_stats["accepted_docs"][lang] = code_stats["accepted_docs"].get(lang, 0) + 1
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1
                if args.max_docs and accepted >= args.max_docs:
                    reached_limit = True
                    break
            if reached_limit:
                break
    (output.parent / "filter_stats.json").write_text(
        json.dumps(filter_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output.parent / "code_language_stats.json").write_text(
        json.dumps(code_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
