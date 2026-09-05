#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model> [--viz|--visualize]"
  echo "Models: bevdepth | lss | bevformer"
  echo "Optional: EXECUTE=1 $0 <model>"
  echo "Optional: VISUALIZE=1 $0 bevdepth"
  exit 1
fi

MODEL="$1"
MODEL_CFG="configs/models/${MODEL}.yaml"
EXECUTE_FLAG=""
VISUALIZE_FLAG=""

if [[ "${EXECUTE:-0}" == "1" ]]; then
  EXECUTE_FLAG="--execute"
fi

if [[ "${VISUALIZE:-0}" == "1" || "${2:-}" == "--viz" || "${2:-}" == "--visualize" ]]; then
  VISUALIZE_FLAG="--visualize"
fi

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "Unknown model config: $MODEL_CFG"
  exit 1
fi

PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" python -m bevbench.benchmark \
  --base-config configs/base.yaml \
  --model-config "$MODEL_CFG" \
  $EXECUTE_FLAG \
  $VISUALIZE_FLAG \
  --out "results/${MODEL}_$(date +%Y%m%d_%H%M%S).json"
