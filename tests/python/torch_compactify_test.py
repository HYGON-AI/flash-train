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


@pytest.mark.parametrize(
    "width", [1, 17, 255, 256, 257, 511, 512, 513, 1023, 2048, 4097]
)
def test_preserves_order_beyond_block_size(width):
    values = torch.arange(width, device="cuda", dtype=torch.int32)
    indices = torch.stack(
        (values, torch.full_like(values, -1), values.masked_fill(values % 3 == 0, -5))
    )
    output, lengths = flash_train.torch.compactify(indices)
    torch.testing.assert_close(lengths, (indices >= 0).sum(-1).int(), rtol=0, atol=0)
    for row in range(3):
        expected = indices[row][indices[row] >= 0]
        torch.testing.assert_close(
            output[row, : expected.numel()], expected, rtol=0, atol=0
        )
