#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# % 2 != 0 )); then
  echo "Usage: $0 IN1 OUT1 [IN2 OUT2 ...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
exec "${HAND_PROCESS_PYTHON:-python}" -m hand_process.launcher server "$@"
