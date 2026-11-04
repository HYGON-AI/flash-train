#!/usr/bin/env bash
set -eo pipefail
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
source "${DTK_ROOT}/env.sh"
set -u
cd "${GITHUB_WORKSPACE:?}"
export PYTHON="${PYTHON:-python3}"
export CI_WORK_DIR="${RUNNER_TEMP:?}/flash-train-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}-${DTK_VER:?}-${PYTHON_VERSION:?}-${TORCH_VERSION:?}"
export WHEEL_OUTPUT="${GITHUB_WORKSPACE}/output/dtk${DTK_VER}/py${PYTHON_VERSION}/torch${TORCH_VERSION}"
export PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10
: "${PYPI_URL:?Set PYPI_URL}"
export PIP_INDEX_URL="${PYPI_URL%/}/nightly/dtk2604/+simple/"
export TORCH_INDEX_URL="${PYPI_URL%/}/nightly/dtk${DTK_VER}/+simple/"
export PIP_TRUSTED_HOST
PIP_TRUSTED_HOST=$("$PYTHON" -c 'import os,urllib.parse; print(urllib.parse.urlparse(os.environ["PYPI_URL"]).hostname)')
case "$DTK_VER" in
  2604) export AICC_MAJOR_VERSION=1.0.0 ;;
  2610) export AICC_MAJOR_VERSION=2.0.0
        export TORCH_INDEX_URL="${PYPI_URL%/}/das_nightly/dtk2610/+simple/" ;;
  *) echo "Unsupported DTK: $DTK_VER" >&2; exit 1 ;;
esac
mkdir -p "$CI_WORK_DIR" "$WHEEL_OUTPUT"
