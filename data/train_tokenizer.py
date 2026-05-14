from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|doc_sep|>",
]


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="训练项目内 BPE tokenizer")
    parser.add_argument("--input", default="data/metadata/accepted_samples.jsonl")
    parser.add_argument("--output", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_texts(Path(args.input)), trainer=trainer)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    stats_path = out.parent.parent / "metadata" / "tokenizer_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "type": "BPE",
                "vocab_size": tokenizer.get_vocab_size(),
                "special_tokens": SPECIAL_TOKENS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
