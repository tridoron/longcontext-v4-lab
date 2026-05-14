#!/usr/bin/env bash
set -euo pipefail
uv run python -m longcontext.train.pretrain --config configs/main_350m_full.yaml --dry-run
