#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model> [nuscenes|truckscenes] [--viz|--visualize]"
  echo "Models: bevdepth | lss | bevformer"
  echo "Datasets: nuscenes (default) | truckscenes"
  echo "Optional: EXECUTE=1 $0 <model>"
  echo "Optional: VISUALIZE=1 $0 bevdepth"
  exit 1
fi

MODEL="$1"
shift

DATASET="${DATASET:-nuscenes}"
EXECUTE_FLAG=""
VISUALIZE_FLAG=""

for arg in "$@"; do
  case "$arg" in
    nuscenes|truckscenes)
      DATASET="$arg"
      ;;
    --viz|--visualize)
      VISUALIZE_FLAG="--visualize"
      ;;
  esac
done

BASE_CFG="configs/base.yaml"
MODEL_CFG="configs/models/${MODEL}.yaml"

if [[ "$DATASET" == "truckscenes" ]]; then
  BASE_CFG="configs/datasets/truckscenes.yaml"
  MODEL_CFG="configs/models/truckscenes/${MODEL}.yaml"
fi

if [[ "${EXECUTE:-0}" == "1" ]]; then
  EXECUTE_FLAG="--execute"
fi

if [[ "${VISUALIZE:-0}" == "1" ]]; then
  VISUALIZE_FLAG="--visualize"
fi

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "Unknown model config: $MODEL_CFG"
  exit 1
fi

PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" python -m bevbench.benchmark \
  --base-config "$BASE_CFG" \
  --model-config "$MODEL_CFG" \
  $EXECUTE_FLAG \
  $VISUALIZE_FLAG \
  --out "results/${MODEL}_${DATASET}_$(date +%Y%m%d_%H%M%S).json"
