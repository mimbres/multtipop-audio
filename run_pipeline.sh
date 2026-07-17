#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/work/miniforge/envs/multtipop_audio/bin/python}"

# The managed kt shell exports a Python 3.12 user-package path globally. Keep
# this Python 3.11 environment isolated and deterministic.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run_pipeline.py" "$@"
