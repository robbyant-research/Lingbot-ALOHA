#!/usr/bin/env bash
# Modified 2026-10-22: activate the repository-local uv environment.
OPENPI_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$OPENPI_PROJECT_DIR/.venv/bin/activate" ]]; then
    printf '%s\n' 'Environment not found. Run bash env_setup/setup_env.sh first.' >&2
    return 1 2>/dev/null || exit 1
fi
source "$OPENPI_PROJECT_DIR/.venv/bin/activate"
export OPENPI_PYTHON="$OPENPI_PROJECT_DIR/.venv/bin/python"
