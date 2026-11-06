#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Reject Flux branding outside the explicit upstream-baseline allowlist."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"flux", re.IGNORECASE)
EXEMPTED_PATHS = {
    # Records the imported backend's provenance.
    "docs/adr/0001-hygon-distributed-backend.md",
    # Intentionally imports and labels the upstream direct API for A/B tests.
    "tests/python/benchmark_flux_family_ops.py",
    # This checker must name the token it rejects.
    "tools/check_hygon_branding.py",
}
EXEMPTED_PREFIXES = (
    # Assembly source names and ELF symbols are the historical loader ABI.
    # The built asset filenames and public flash-train API use Hygon names.
    "src/primitive/hygon/distributed/vendor/src/ag_gemm/gemm_kernels/",
)


def tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    )
    return sorted(line for line in output.splitlines() if line)


def find_violations() -> list[str]:
    violations: list[str] = []
    for path in tracked_files():
        if path in EXEMPTED_PATHS:
            continue
        if path.startswith(EXEMPTED_PREFIXES):
            continue
        file_path = REPO_ROOT / path
        if TOKEN.search(path):
            violations.append(f"{path}: Flux-branded tracked pathname")
            continue
        try:
            payload = file_path.read_bytes()
        except OSError as error:
            violations.append(f"{path}: unable to read: {error}")
            continue
        if b"\0" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if TOKEN.search(line):
                violations.append(f"{path}:{line_number}:{line.strip()}")
    return violations


def main() -> int:
    argparse.ArgumentParser().parse_args()
    violations = find_violations()
    if violations:
        print("Unexpected Flux branding outside the allowlist:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("Hygon branding check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
