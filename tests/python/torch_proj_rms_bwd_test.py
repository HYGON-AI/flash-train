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
    grad_proj = (
        torch.rand(expected_proj.shape, device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    grad_r = (
        torch.rand(expected_r.shape, device="cuda", generator=generator) * 0.02 - 0.01
    ).to(dtype)
    expected_x, expected_weight = torch.autograd.grad(
        (expected_proj, expected_r), (rx, rw), (grad_proj.float(), grad_r.float())
    )
    actual_x, actual_weight = flash_train.torch.proj_rms_bwd(
        grad_proj, grad_r, x, weight, expected_r.detach().to(dtype)
    )
    rtol, atol = TOLERANCES[dtype]
    assert torch.any(
        expected_x.abs() > atol + rtol * expected_x.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual_x, expected_x.to(dtype), rtol=rtol, atol=atol)
    rtol, atol = TOLERANCES[weight_dtype]
    assert torch.any(
        expected_weight.abs() > atol + rtol * expected_weight.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(
        actual_weight, expected_weight.to(weight_dtype), rtol=rtol, atol=atol
    )
