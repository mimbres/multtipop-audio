#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/.venv}"
PYTHON="${ENV_PREFIX}/bin/python"
FLASHSR_DIR="${PROJECT_ROOT}/third_party/FlashSR_Inference"
FLASHSR_COMMIT="2292814a7ef74f61a5479c8d96e653d2f90f369d"
NODE_RUNTIME_DIR="${PROJECT_ROOT}/third_party/node_runtime"
NODE_BIN="${NODE_RUNTIME_DIR}/node_modules/node/bin/node"

# Keep the Python 3.11 environment isolated from shell-level package overrides.
unset PIP_CONSTRAINT
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  printf 'Conda was not found. Install Conda or set CONDA_BIN.\n' >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.11 pip
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install \
  torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
"${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

# Install a project-local Node.js 22 runtime for recent yt-dlp versions.
if [[ ! -x "${NODE_BIN}" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    printf 'npm is required to install the yt-dlp Node.js 22 runtime.\n' >&2
    exit 1
  fi
  npm install --prefix "${NODE_RUNTIME_DIR}" --no-save --no-package-lock node@22
fi
"${NODE_BIN}" --version

if [[ ! -d "${FLASHSR_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${FLASHSR_DIR}")"
  git clone https://github.com/jakeoneijk/FlashSR_Inference.git "${FLASHSR_DIR}"
fi
git -C "${FLASHSR_DIR}" fetch origin "${FLASHSR_COMMIT}"
git -C "${FLASHSR_DIR}" checkout --detach "${FLASHSR_COMMIT}"

# The pipeline imports the pinned source tree directly. Avoid installing its
# training-only dependency metadata into this inference environment.

printf 'Environment ready. Python: %s\n' "${PYTHON}"
