from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class PackedMemmapDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int, pad_token_id: int = 0) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.sample_len = seq_len + 1
        self.pad_token_id = pad_token_id
        self.tokens = np.memmap(self.path, mode="r", dtype=np.uint32)
        if self.tokens.size % self.sample_len != 0:
            raise ValueError(f"{self.path} 长度不是 seq_len+1 的整数倍")
        self.num_samples = self.tokens.size // self.sample_len

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.sample_len
        row = np.asarray(self.tokens[start : start + self.sample_len], dtype=np.int64)
        input_ids = torch.from_numpy(row[:-1].copy()).long()
        labels = torch.from_numpy(row[1:].copy()).long()
        attention_mask = input_ids.ne(self.pad_token_id)
        labels = labels.masked_fill(labels.eq(self.pad_token_id), IGNORE_INDEX)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
