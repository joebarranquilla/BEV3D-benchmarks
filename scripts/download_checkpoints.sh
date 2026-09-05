#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

cat << 'EOF'
Checkpoint download helper

Option A (recommended): add URLs in configs/checkpoint_urls.env then run this script.
Option B: copy-paste manual wget commands shown below.

Expected files:
  checkpoints/bevdepth_r50.pth
  checkpoints/lss_r50.pth
  checkpoints/bevformer_base.pth

Manual template:
  wget -O checkpoints/bevdepth_r50.pth "<BEVDEPTH_CHECKPOINT_URL>"
  wget -O checkpoints/lss_r50.pth "<LSS_CHECKPOINT_URL>"
  wget -O checkpoints/bevformer_base.pth "<BEVFORMER_CHECKPOINT_URL>"
EOF

ENV_FILE="$ROOT_DIR/configs/checkpoint_urls.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"

  [[ -n "${BEVDEPTH_CKPT_URL:-}" ]] && wget -O "$CKPT_DIR/bevdepth_r50.pth" "$BEVDEPTH_CKPT_URL"
  [[ -n "${LSS_CKPT_URL:-}" ]] && wget -O "$CKPT_DIR/lss_r50.pth" "$LSS_CKPT_URL"
  [[ -n "${BEVFORMER_CKPT_URL:-}" ]] && wget -O "$CKPT_DIR/bevformer_base.pth" "$BEVFORMER_CKPT_URL"

  echo "Downloads attempted from $ENV_FILE"
else
  echo "No $ENV_FILE found. Create it from configs/checkpoint_urls.example.env"
fi
