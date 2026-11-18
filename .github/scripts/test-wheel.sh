#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$WHEEL_OUTPUT"
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: No visible HCU; refusing to skip all correctness tests')
print(f'Visible HCU devices: {torch.cuda.device_count()}')
PY
cpp_rc=0
if [[ "${CPP_TESTS:-false}" == true ]]; then
  "$GITHUB_WORKSPACE/build-wheel/bin/ftrain_test" --gtest_brief=1 --gtest_shuffle \
    --gtest_output="xml:$WHEEL_OUTPUT/cpp_tests.xml" || cpp_rc=$?
fi
shopt -s nullglob
wheels=("$WHEEL_OUTPUT"/*.whl)
[[ ${#wheels[@]} == 1 ]] || { echo 'Expected exactly one repaired wheel' >&2; exit 1; }
"$PYTHON" -m pip install --no-deps --force-reinstall "${wheels[0]}"
cd /tmp
python_rc=0
"$PYTHON" -m pytest "$GITHUB_WORKSPACE/tests/python" --junitxml="$WHEEL_OUTPUT/python_tests.xml" || python_rc=$?
if ((cpp_rc != 0 || python_rc != 0)); then
  echo "Tests failed: C++ exit=$cpp_rc; Python exit=$python_rc" >&2
  exit 1
fi
