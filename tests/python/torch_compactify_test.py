# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

SHAPE_GRID = [(1, 3), (7, 17), (33, 64)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
def test_matches_torch(shape):
    generator = torch.Generator(device="cuda").manual_seed(7)
    indices = torch.randint(
        -1, 32, shape, device="cuda", generator=generator, dtype=torch.int32
    )
    indices[0] = -1
    expected_length = (indices >= 0).sum(-1).to(torch.int32)
    actual, length = flash_train.torch.compactify(indices)
    # Only the valid prefix is defined; the tail is not an output contract.
    torch.testing.assert_close(length, expected_length, rtol=0, atol=0)
    for row in range(shape[0]):
        expected = indices[row][indices[row] >= 0]
        torch.testing.assert_close(
            actual[row, : expected.numel()], expected, rtol=0, atol=0
        )
