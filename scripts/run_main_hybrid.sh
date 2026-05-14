#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
uv run python -m longcontext.train.pretrain --config configs/main_350m_hybrid.yaml --dry-run
