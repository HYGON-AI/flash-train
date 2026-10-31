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

SHAPE_GRID = [(1, 2, 3), (7, 4, 5), (33, 8, 3), (1025, 4, 5), (7, 4, 20), (7, 8, 4)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_torch(shape, dtype):
    batch, size, iterations = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = (
        torch.rand((batch, size, size), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    rx = logits.detach().float().requires_grad_()
    initial = rx.softmax(-1) + 1e-6
    expected = initial
    for _ in range(iterations):
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(-2, keepdim=True) + 1e-6)
    grad = (
        torch.rand(logits.shape, device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    expected_grad = torch.autograd.grad(expected, rx, grad.float())[0]
    actual = flash_train.torch.sinkhorn_bwd(grad, initial.detach(), iterations)
    rtol, atol = TOLERANCES[dtype]
    assert torch.any(
        expected_grad.abs() > atol + rtol * expected_grad.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual, expected_grad.to(dtype), rtol=rtol, atol=atol)
