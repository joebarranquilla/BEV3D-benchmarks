#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/nuscenes"

mkdir -p "$DATA_DIR"

echo "This script prepares the local directory for nuScenes-mini."
echo "Due to nuScenes licensing, download files manually from the official portal:"
echo "https://www.nuscenes.org/download"
echo
echo "Expected archive example: v1.0-mini.tgz"
echo "Then run:"
echo "  tar -xzf /path/to/v1.0-mini.tgz -C $DATA_DIR"
echo
echo "Current target directory: $DATA_DIR"
