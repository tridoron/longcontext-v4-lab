#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/main_350m_hybrid.yaml}
WEIGHTS=${2:-}
WEIGHT_ARGS=()
if [[ -n "$WEIGHTS" ]]; then
  WEIGHT_ARGS=(--weights "$WEIGHTS")
fi
uv run python -m longcontext.eval.eval_kv_cache --config "$CONFIG"
uv run python -m longcontext.eval.eval_speed --config "$CONFIG" "${WEIGHT_ARGS[@]}" --lengths 1024 4096 --steps 1
