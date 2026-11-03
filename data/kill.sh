#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
exec "${HAND_PROCESS_PYTHON:-python}" -m hand_process.service_control \
  --project-root "${PROJECT_ROOT}" "$@"
