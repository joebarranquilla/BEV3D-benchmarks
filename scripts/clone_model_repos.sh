#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
ENV_FILE="$ROOT_DIR/configs/model_repo_urls.env"

mkdir -p "$THIRD_PARTY_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Create $ENV_FILE from configs/model_repo_urls.example.env first."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -n "${BEVDEPTH_REPO_URL:-}" && ! -d "$THIRD_PARTY_DIR/BEVDepth" ]]; then
  git clone "$BEVDEPTH_REPO_URL" "$THIRD_PARTY_DIR/BEVDepth"
fi
if [[ -n "${LSS_REPO_URL:-}" && ! -d "$THIRD_PARTY_DIR/LSS" ]]; then
  git clone "$LSS_REPO_URL" "$THIRD_PARTY_DIR/LSS"
fi
if [[ -n "${BEVFORMER_REPO_URL:-}" && ! -d "$THIRD_PARTY_DIR/BEVFormer" ]]; then
  git clone "$BEVFORMER_REPO_URL" "$THIRD_PARTY_DIR/BEVFormer"
fi

echo "Repository clone step complete."
