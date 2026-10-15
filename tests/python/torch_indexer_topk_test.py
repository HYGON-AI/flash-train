# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

SHAPE_GRID = [(3, 17, 2), (7, 33, 4), (33, 64, 8)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
def test_matches_torch(shape):
    rows, columns, top_k = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Distinct values avoid imposing a tie-breaking rule on equal elements.
    values = torch.stack(
        [
            torch.randperm(columns, device="cuda", generator=generator).float()
            for _ in range(rows)
        ]
    )
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    expected_values, expected_indices = values.topk(top_k, sorted=True)
    actual_indices, actual_values = flash_train.torch.indexer_topk(
        values, lengths, top_k, next_n=1, order_values=True
    )
    expected_indices = expected_indices.to(torch.int32)
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
