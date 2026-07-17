#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

# Keep the project environment isolated from user-level Python packages.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment not found. Run setup_env.sh or set PYTHON_BIN.\n' >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m multtipop_audio.pipeline "$@"
