#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

WORLD_SIZE="${FTRAIN_HYGON_TEST_WORLD_SIZE:-8}"
if [[ "$WORLD_SIZE" -lt 2 ]]; then
  echo "FTRAIN_HYGON_TEST_WORLD_SIZE must be at least 2 (got $WORLD_SIZE)" >&2
  exit 2
fi

TEST_FILE="tests/python/test_hygon_family_ops.py"

AG_CASES=(
  native_nk_a2a
  native_kn_a2a
  native_kn_ring_fixed
  native_kn_ring
  native_kn_ring_readymask
)

MOE_CASES=(
  "a2a-nk_direct-False"
  "a2a-kn_direct-True"
  "a2a_ragged_signal-nk_direct-False"
  "a2a_ragged_signal-kn_direct-True"
)

for route in "${AG_CASES[@]}"; do
  echo "==> AGGroupGemm ${route} (world=${WORLD_SIZE})"
  torchrun --standalone --nproc-per-node="$WORLD_SIZE" \
    -m pytest -q -s \
    "${TEST_FILE}::test_formal_ag_group_gemm[${route}]"
done

for case in "${MOE_CASES[@]}"; do
  echo "==> MoEReduceRS ${case} (world=${WORLD_SIZE})"
  torchrun --standalone --nproc-per-node="$WORLD_SIZE" \
    -m pytest -q -s \
    "${TEST_FILE}::test_formal_moe_reduce_rs[${case}]"
done
