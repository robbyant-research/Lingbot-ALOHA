#!/usr/bin/env bash
# Modified 2026-10-22: install the local environment with the upstream OpenPI uv workflow.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e .
