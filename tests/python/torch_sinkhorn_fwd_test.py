# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

DTYPE = torch.bfloat16


def make_sinkhorn_inputs(batch=4, size=8, seed=41):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(batch, size, size, device="cuda", generator=generator).to(DTYPE)


def test_binding_rejects_empty_sinkhorn_without_dividing_by_zero():
    with pytest.raises(RuntimeError, match="positive width"):
        ft.sinkhorn_fwd(torch.empty(2, 0, 0, device="cuda"), 5)


@pytest.mark.parametrize(
    "shape", [(4, 4), (2, 4, 4), (1, 2, 4, 4), (1, 1, 2, 4, 4), (1, 1, 1, 2, 4, 4)]
)
def test_arbitrary_batch_rank_matches_torch_forward_and_backward(shape):
    torch.manual_seed(189)
    logits = torch.randn(shape, device="cuda", requires_grad=True)
    # Sinkhorn starts from a row softmax, then alternates row/column normalization.
    expected = torch.softmax(logits, dim=-1) + 1e-6
    for _ in range(3):
        expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(dim=-2, keepdim=True) + 1e-6)
    grad = torch.randn_like(logits)
    expected.backward(grad)
    actual, state = ft.sinkhorn_fwd(logits.detach(), 3)
    actual_grad = ft.sinkhorn_bwd(grad, state.reshape(shape), 3)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual_grad, logits.grad, rtol=1e-4, atol=1e-5)


def test_empty_batch_is_noop_in_both_directions():
    logits = torch.empty(1, 0, 1, 4, 4, device="cuda")
    actual, state = ft.sinkhorn_fwd(logits, 3)
    grad = ft.sinkhorn_bwd(torch.empty_like(logits), state, 3)
    torch.cuda.synchronize()
    assert actual.shape == logits.shape
    assert state.shape == (0, 4, 4)
    assert grad.shape == logits.shape


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
    actual, state = flash_train.torch.sinkhorn_fwd(logits, iterations)
    rtol, atol = TOLERANCES[dtype]
    assert torch.any(
        expected.abs() > atol + rtol * expected.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(state, initial.detach(), rtol=2e-4, atol=2e-5)
