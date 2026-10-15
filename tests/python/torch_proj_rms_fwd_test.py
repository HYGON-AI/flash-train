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

SHAPE_GRID = [(1, 8, 32), (7, 16, 64), (33, 24, 128)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", DTYPES)
def test_matches_torch(shape, dtype, weight_dtype):
    rows, outputs, reduction = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = (
        torch.rand((rows, reduction), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    weight = (
        torch.rand((outputs, reduction), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(weight_dtype)
    rx = x.detach().float().requires_grad_()
    rw = weight.detach().float().requires_grad_()
    expected_proj = rx @ rw.T
    expected_r = 1 / (rx.square().mean(-1, keepdim=True).sqrt() + 1e-6)
    actual_proj, actual_r = flash_train.torch.proj_rms_fwd(x, weight)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(
        actual_proj, expected_proj.to(dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(actual_r, expected_r.to(dtype), rtol=rtol, atol=atol)
