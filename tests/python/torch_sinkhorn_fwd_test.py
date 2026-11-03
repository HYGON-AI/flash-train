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
    # Megatron counts the row softmax as the first row normalization.
    expected = torch.softmax(logits, dim=-1) + 1e-6
    for step in range(3):
        if step > 0:
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
    for step in range(iterations):
        if step > 0:
            expected = expected / (expected.sum(-1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(-2, keepdim=True) + 1e-6)
    actual, state = flash_train.torch.sinkhorn_fwd(logits, iterations)
    rtol, atol = TOLERANCES[dtype]
    assert torch.any(
        expected.abs() > atol + rtol * expected.abs()
    ), "reference cannot reject all-zero output"
    torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(state, initial.detach(), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [4, 8])
@pytest.mark.parametrize("iterations", [3, 5, 7, 10, 15, 20])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_wave_forward_backward_replay_matches_autograd(dtype, size, iterations, eps):
    # The partial final block and four matrices per wave exercise subgroup
    # isolation. Larger eps rejects an extra first row normalization.
    torch.manual_seed(761)
    base = torch.randn(1025, size, size, device="cuda").to(dtype)
    incoming = (torch.randn_like(base) * 0.5).to(dtype)
    logits, grad = base.clone(), incoming.clone()
    _, state = flash_train.torch.sinkhorn_fwd(logits, iterations, eps)
    flash_train.torch.sinkhorn_bwd(grad, state, iterations, eps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, state = flash_train.torch.sinkhorn_fwd(logits, iterations, eps)
        actual_grad = flash_train.torch.sinkhorn_bwd(grad, state, iterations, eps)
    rtol, atol = TOLERANCES[dtype]
    for scale in (1.0, -0.75):
        logits.copy_(base * scale)
        grad.copy_(incoming * scale)
        rx = logits.float().detach().requires_grad_()
        initial = rx.softmax(-1) + eps
        expected = initial
        for step in range(iterations):
            if step > 0:
                expected = expected / (expected.sum(-1, keepdim=True) + eps)
            expected = expected / (expected.sum(-2, keepdim=True) + eps)
        expected_grad = torch.autograd.grad(expected, rx, grad.float())[0]
        graph.replay()
        assert expected_grad.abs().max() > atol
        torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
        torch.testing.assert_close(state, initial.detach(), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual_grad, expected_grad.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("iterations", [1, 3])
def test_initial_softmax_counts_as_first_row_normalization(dtype, size, iterations):
    # Megatron starts with softmax+eps then column normalization. This is an
    # independent autograd reference; eps=0.5 rejects the former extra row step.
    torch.manual_seed(910)
    logits = (torch.randn(33, size, size, device="cuda") * 0.3).to(dtype)
    rx = logits.float().detach().requires_grad_()
    eps = 0.5
    initial = rx.softmax(-1) + eps
    expected = initial / (initial.sum(-2, keepdim=True) + eps)
    extra_row = initial / (initial.sum(-1, keepdim=True) + eps)
    extra_row = extra_row / (extra_row.sum(-2, keepdim=True) + eps)
    assert (expected - extra_row).abs().max() > 1e-3
    for _ in range(iterations - 1):
        expected = expected / (expected.sum(-1, keepdim=True) + eps)
        expected = expected / (expected.sum(-2, keepdim=True) + eps)
    incoming = torch.randn_like(logits) * 128
    expected.backward(incoming.float())
    actual, state = ft.sinkhorn_fwd(logits, iterations, eps)
    gradient = ft.sinkhorn_bwd(incoming, state, iterations, eps)
    rtol, atol = TOLERANCES[dtype]
    assert rx.grad.abs().max() > atol
    torch.testing.assert_close(state, initial.detach(), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(gradient, rx.grad.to(dtype), rtol=rtol, atol=atol)
