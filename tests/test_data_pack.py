from __future__ import annotations

import numpy as np

from data.dataset import PackedMemmapDataset
from data.pack_dataset import pack_stream
from data.prepare_corpus import doc_hash, normalize_text
from data.split_validation import split_validation


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
