#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" - <<'PY'
import os,sys,sysconfig
from pathlib import Path
assert f"{sys.version_info.major}.{sys.version_info.minor}" == os.environ['PYTHON_VERSION'], sys.version
assert Path(sysconfig.get_path('include'), 'Python.h').is_file()
PY
: "${AICC_INSTALLER_URL:=${RESOURCE_SERVER_URL:+${RESOURCE_SERVER_URL%/}/ai_cc/Nightly/hcu_llvm_installer.sh}}"
: "${AICC_INSTALLER_URL:?Set AICC_INSTALLER_URL or RESOURCE_SERVER_URL}"
curl -fsSL --retry 3 --retry-delay 2 -o "$CI_WORK_DIR/hcu_llvm_installer.sh" "$AICC_INSTALLER_URL"
if [[ -n "${CI_REPRO_DIR:-}" && -f "$CI_REPRO_DIR/manifest.json" ]]; then
  # Resolve the nightly selector once. Save its exact URL/checksum before
  # installing so an installation failure still leaves useful metadata.
  mkdir -p "$CI_WORK_DIR/aicc" "$CI_REPRO_DIR"
  bash "$CI_WORK_DIR/hcu_llvm_installer.sh" --major "$AICC_MAJOR_VERSION" \
    --dtk-dir "$DTK_ROOT" --dry-run | tee "$CI_REPRO_DIR/aicc-selection.txt"
  mapfile -t aicc_selection < <("$PYTHON" - "$CI_REPRO_DIR/aicc-selection.txt" <<'PY'
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
text = Path(sys.argv[1]).read_text()
version = re.search(r'^完整版本:\s*(\S+)\s*$', text, re.MULTILINE)
url = next((u for u in re.findall(r'https?://\S+', text) if urlsplit(u).path.endswith('.run')), None)
if not version or not url:
    raise SystemExit('Cannot resolve exact AICC version/package from installer output')
print(version[1])
print(url)
print(Path(unquote(urlsplit(url).path)).name)
PY
  )
  [[ ${#aicc_selection[@]} == 3 ]] || { echo 'Cannot capture exact AICC package' >&2; exit 1; }
  curl -fSL --retry 3 --retry-delay 2 -o "$CI_WORK_DIR/aicc/${aicc_selection[2]}" "${aicc_selection[1]}"
  "$PYTHON" "$GITHUB_WORKSPACE/.github/scripts/capture-ci-env.py" aicc \
    --selection "$CI_REPRO_DIR/aicc-selection.txt" --package "$CI_WORK_DIR/aicc/${aicc_selection[2]}"
  (
    cd "$CI_WORK_DIR/aicc"
    bash "$CI_WORK_DIR/hcu_llvm_installer.sh" --version "${aicc_selection[0]}" --dtk-dir "$DTK_ROOT"
  )
else
  bash "$CI_WORK_DIR/hcu_llvm_installer.sh" --major "$AICC_MAJOR_VERSION" --dtk-dir "$DTK_ROOT"
fi
test -f "$DTK_ROOT/hcu_llvm.info"
printf 'torch==%s\nnumpy==%s\n' "$TORCH_VERSION" "$NUMPY_VERSION" > "$CI_WORK_DIR/constraints.txt"
"$PYTHON" -m pip install --force-reinstall --no-deps --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
"$PYTHON" -m pip install --constraint "$CI_WORK_DIR/constraints.txt" "torch==$TORCH_VERSION" "numpy==$NUMPY_VERSION" pytest ninja wheel pybind11 'scikit-build-core>=1.0' 'cmake>=3.29' ciupload
"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import os
from importlib.metadata import version
assert version('torch').split('+',1)[0] == os.environ['TORCH_VERSION'], version('torch')
assert f"dtk{os.environ['DTK_VER']}" in version('torch'), version('torch')
assert version('numpy') == os.environ['NUMPY_VERSION'], version('numpy')
print(f"Torch: {version('torch')}; NumPy: {version('numpy')}")
PY
