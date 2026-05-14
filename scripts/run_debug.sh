#!/usr/bin/env bash
set -euo pipefail
uv run python -m longcontext.train.pretrain --config configs/debug_120m_full.yaml --dry-run
