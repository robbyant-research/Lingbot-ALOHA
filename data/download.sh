#!/usr/bin/env bash
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: $0 [CONFIG]" >&2
  exit 2
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
exec "${HAND_PROCESS_PYTHON:-python}" -m hand_process.cli assets-download \
  --config "${1:-configs/server.yaml}" --manifest "${PROJECT_ROOT}/model_assets.json"
