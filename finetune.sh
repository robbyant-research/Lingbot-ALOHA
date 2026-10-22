#!/usr/bin/env bash
# Modified 2026-10-21: require a retained config and remove experiment-specific launch settings.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME=${1:?Usage: bash finetune.sh CONFIG_NAME --exp-name EXP_NAME [training options]}
shift
cd "$PROJECT_DIR"
exec python scripts/train.py "$CONFIG_NAME" "$@"
