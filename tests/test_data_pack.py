from __future__ import annotations

import pytest
import numpy as np

from data.dataset import PackedMemmapDataset
from data.pack_dataset import pack_stream
from data.prepare_corpus import doc_hash, normalize_text, stream_regular_source, validate_sample
from data.split_validation import split_validation, write_val_doc_hashes
from data.tokenize_corpus import assert_not_validation_doc, load_val_doc_hashes


def test_normalize_and_hash_stable():
    text = normalize_text(" a\r\n\r\n\r\nb\x00 ")
    assert text == "a\n\nb"
    assert doc_hash("web", text) == doc_hash("web", text)


def test_pack_and_mmap_dataset(tmp_path):
    src = tmp_path / "tokens.bin"
    np.arange(20, dtype=np.uint32).tofile(src)
    out = tmp_path / "packed.bin"
    count = pack_stream([src], out, seq_len=4, pad_id=0)
    assert count == 4
    dataset = PackedMemmapDataset(out, seq_len=4)
    item = dataset[0]
    assert item["input_ids"].tolist() == [0, 1, 2, 3]
    assert item["labels"].tolist() == [1, 2, 3, 4]


def test_padded_tail_labels_are_ignored(tmp_path):
    src = tmp_path / "tokens.bin"
    np.arange(7, dtype=np.uint32).tofile(src)
    out = tmp_path / "packed.bin"

    count = pack_stream([src], out, seq_len=4, pad_id=0)
    dataset = PackedMemmapDataset(out, seq_len=4)
    item = dataset[1]

    assert count == 2
    assert item["input_ids"].tolist() == [5, 6, 0, 0]
    assert item["labels"].tolist() == [6, -100, -100, -100]
    assert item["attention_mask"].tolist() == [True, True, False, False]


def test_split_validation_doc_hash_exclusion():
    rows = []
    for source in ["web", "code"]:
        for idx in range(4):
            text = f"{source}-{idx}"
            rows.append(
                {
                    "source": source,
                    "doc_hash": doc_hash(source, text),
                    "text": text,
                    "approx_tokens": 10,
                }
            )
    train, validation, stats = split_validation(
        rows,
        {"val_pretrain": {"web": 15, "code": 10}},
    )
    val_hashes = {row["doc_hash"] for row in validation["val_pretrain"]}
    assert val_hashes
    assert all(row["doc_hash"] not in val_hashes for row in train)
    assert stats["validation_doc_hashes"] == len(val_hashes)


def test_split_validation_raises_when_quota_cannot_be_filled():
    rows = [
        {
            "source": "web",
            "doc_hash": doc_hash("web", "tiny"),
            "text": "tiny",
            "approx_tokens": 10,
        }
    ]

    with pytest.raises(RuntimeError, match="validation quota 未填满"):
        split_validation(rows, {"val_pretrain": {"web": 20}})


def test_val_doc_hashes_written_before_train_tokenization_checks(tmp_path):
    validation = {"val_pretrain": [{"doc_hash": "abc"}]}
    val_hashes_path = tmp_path / "val_doc_hashes.txt"

    hashes = write_val_doc_hashes(val_hashes_path, validation)

    assert hashes == {"abc"}
    assert load_val_doc_hashes(val_hashes_path) == {"abc"}
    with pytest.raises(RuntimeError, match="训练 shard 包含验证集文档"):
        assert_not_validation_doc({"doc_hash": "abc"}, hashes, line_no=7)


def test_missing_val_doc_hashes_rejects_train_tokenization(tmp_path):
    with pytest.raises(FileNotFoundError, match="训练 shard 构建前必须先生成"):
        load_val_doc_hashes(tmp_path / "missing.txt")


class FakeTokenizer:
    def encode(self, text: str):
        return type("Encoded", (), {"ids": text.split()})()


def repeated_lines(words: int, lines: int) -> str:
    line = " ".join(f"token{i}" for i in range(words // lines))
    return "\n".join(line for _ in range(lines))


def test_prepare_corpus_records_source_specific_truncation_limits():
    seen_hashes: set[str] = set()
    web_row, web_reason = validate_sample(
        {"text": repeated_lines(9000, 30)},
        "web",
        seen_hashes,
        tokenizer=FakeTokenizer(),
    )
    long_row, long_reason = validate_sample(
        {"text": repeated_lines(70000, 70)},
        "long_doc",
        seen_hashes,
        tokenizer=FakeTokenizer(),
    )

    assert web_reason is None
    assert long_reason is None
    assert web_row is not None
    assert long_row is not None
    assert web_row["max_tokens"] == 8192
    assert long_row["max_tokens"] == 65536
    assert web_row["approx_tokens"] > web_row["max_tokens"]
    assert long_row["approx_tokens"] > long_row["max_tokens"]


def test_prepare_corpus_uses_streaming_without_force_redownload(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([])

    monkeypatch.setattr("data.prepare_corpus.load_dataset", fake_load_dataset)

    list(stream_regular_source({"path": "demo/path", "split": "train"}))

    assert calls
    assert calls[0][1]["streaming"] is True
    assert calls[0][1].get("download_mode") != "force_redownload"
