#!/usr/bin/env bash
set -euo pipefail
uv run python -m longcontext.eval.eval_kv_cache --config configs/main_350m_full.yaml
uv run python -m longcontext.eval.eval_kv_cache --config configs/main_350m_hybrid.yaml
