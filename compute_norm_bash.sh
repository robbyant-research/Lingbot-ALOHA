#!/usr/bin/env bash
# Modified 2026-10-21: require a retained config instead of a removed Lingbot preset.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME=${1:?Usage: bash compute_norm_bash.sh CONFIG_NAME [normalization options]}
shift
cd "$PROJECT_DIR"
exec python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" "$@"
