from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pack_stream(input_files: list[Path], output: Path, seq_len: int, pad_id: int = 0) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_len = seq_len + 1
    carry = np.empty((0,), dtype=np.uint32)
    samples = 0
    with output.open("wb") as out_f:
        for path in input_files:
            tokens = np.memmap(path, mode="r", dtype=np.uint32)
            stream = np.concatenate([carry, np.asarray(tokens)])
            usable = (len(stream) // sample_len) * sample_len
            if usable:
                stream[:usable].astype(np.uint32).tofile(out_f)
                samples += usable // sample_len
            carry = stream[usable:]
        if carry.size:
            padded = np.full((sample_len,), pad_id, dtype=np.uint32)
            padded[: min(carry.size, sample_len)] = carry[:sample_len]
            padded.tofile(out_f)
            samples += 1
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 fixed-length LM 样本")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pad-id", type=int, default=0)
    args = parser.parse_args()
    count = pack_stream([Path(p) for p in args.inputs], Path(args.output), args.seq_len, args.pad_id)
    print(f"packed_samples={count}")


if __name__ == "__main__":
    main()
