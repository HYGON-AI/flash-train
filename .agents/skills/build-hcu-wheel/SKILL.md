---
name: build-hcu-wheel
description: Build the flash-train HCU wheel with the repository build_wheel.sh entry point, including optional C++ test binaries and diagnosing missing build prerequisites.
---

# Build flash-train on HCU

Run commands from the repository root in Linux. The root build_wheel.sh forwards to .github/scripts/_build_hcu_wheel.sh, shared by CI and local builds. CI calls the shared script directly.

The environment must already provide Python 3.10+ and development headers, HCU/HIP Torch, DTK and AICC. The script checks DTK env.sh and verifies that aicc can execute --version. Install Python build tools in the selected interpreter before running:

```bash
python3 -m pip install "scikit-build-core>=1.0" "cmake>=3.29" ninja pybind11
PYTHON=python3 bash build_wheel.sh
```

For C++ test binaries or custom environments:

```bash
PYTHON=python3 DTK_ROOT=/opt/dtk bash build_wheel.sh --out-dir dist --build-tests
```

`PYTHON` (default python), `DTK_ROOT` (default /opt/dtk), and optional `FTRAIN_AICC` select the environment. Use `--check-only` to validate without compiling. Paths are relative to the repository root. Select the Torch version in the environment before building; the script does not install or replace Torch/DTK/AICC. A missing prerequisite stops the build. Check the reported interpreter and package versions before repairing the environment.

The script removes the fixed build-wheel CMake cache to avoid mixing Torch versions, explicitly enables/disables C++ tests, and builds in a temporary staging directory before copying the wheel to the output directory. Use a fresh output directory for each build and do not run concurrent builds in one checkout.

The default artifact is dist/flash_train-*.whl. With --build-tests, the C++ executable is build-wheel/bin/ftrain_test. Source the same DTK env.sh before running tests. GPU availability is required for tests, not wheel compilation.

CI is defined in .github/workflows/ci.yml. It repairs the wheel version and archives the artifact. PR and manual runs test the repaired artifacts. As in BoltOPs, develop pushes build without tests and publish those artifacts; do not rebuild in the publish job. PRs and manual runs do not publish.

The active CI matrix is DTK 26.10 / Python 3.12 / Torch 2.11.0 and 2.13.0. C++ tests build/run on Torch 2.11.0; Python tests run on both versions for PR and manual runs. CI selects AICC 2.0.0 and Torch wheels from das_nightly/dtk2610. The nightly/dtk2604 index remains the general CI-tool index, matching BoltOPs; it does not select the Torch build.
