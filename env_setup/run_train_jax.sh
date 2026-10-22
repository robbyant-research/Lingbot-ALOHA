#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT_DIR/env_setup/env.sh"
cd "$PROJECT_DIR"
exec "$OPENPI_PYTHON" scripts/train.py "$@"
