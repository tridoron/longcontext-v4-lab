from __future__ import annotations

import argparse
from dataclasses import asdict

import yaml

from longcontext.eval.common import write_csv
from longcontext.model.config import LongContextConfig
from longcontext.model.kv_cache import estimate_kv_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache 理论占用估算")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--output", default="outputs/artifacts/kv_cache_results.csv")
    args = parser.parse_args()
    config = LongContextConfig.from_dict(yaml.safe_load(open(args.config, encoding="utf-8")))
    rows = [asdict(estimate_kv_cache(config, length)) for length in args.lengths]
    write_csv(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
