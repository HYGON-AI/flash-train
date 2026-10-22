# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

# Include a non-power-of-two row count; reductions stay within launcher coverage.
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# FP32 reduction error plus low-precision storage rounding; absolute error
# also covers near-zero outputs/gradients for inputs bounded to [-0.1, 0.1].
TOLERANCES = {
    torch.float32: (2e-4, 2e-5),
    torch.float16: (2e-2, 2e-3),
    torch.bfloat16: (5e-2, 5e-3),
}

SHAPE_GRID = [(1, 2, 16), (7, 4, 32), (33, 4, 64), (257, 4, 40), (3, 8, 17)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_torch(shape, dtype):
    rows, streams, channels = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = (
        torch.rand((rows, 1, streams, channels), device="cuda", generator=generator)
        * 0.2
        - 0.1
    ).to(dtype)
    h = torch.rand((rows, 1, streams), device="cuda", generator=generator) * 0.2 - 0.1
    grad = (
        torch.rand((rows, 1, channels), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    expected_x = (grad.float().unsqueeze(-2) * h.unsqueeze(-1)).to(dtype)
    expected_h = (grad.float().unsqueeze(-2) * x.float()).sum(-1)
    actual_x, actual_h = flash_train.torch.h_aggregate_bwd(grad, x, h)
    rtol, atol = TOLERANCES[dtype]
    assert torch.any(
        expected_x.abs() > atol + rtol * expected_x.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual_x, expected_x, rtol=rtol, atol=atol)
    assert torch.any(
        expected_h.abs() > atol + rtol * expected_h.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual_h, expected_h, rtol=rtol, atol=atol)
