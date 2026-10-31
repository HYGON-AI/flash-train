# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import flash_train.torch
import flash_train.torch as ft


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def test_workspace_operator_on_current_stream_and_graph_replay():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = rand(64, 32)
        weight = rand(8, 32, dtype=torch.float32)
        alphas = [torch.ones(1, device="cuda") for _ in range(3)]
        bias = torch.zeros(8, device="cuda")
        hp, hpost, hr, r, proj = ft.proj_rms_compute_h_fwd(x, weight, *alphas, bias, 2)
        gradients = [torch.ones_like(t) for t in (hp, hpost, hr, r)]
        args = (*gradients, hp, hpost, proj, r, x, weight, *alphas, 2)
        expected = ft.proj_rms_compute_h_bwd(*args)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = ft.proj_rms_compute_h_bwd(*args)
    for scale in (0.25, -2.0):
        with torch.cuda.stream(stream):
            for gradient in gradients:
                gradient.mul_(scale)
            expected = ft.proj_rms_compute_h_bwd(*args)
            for result in actual:
                result.fill_(float("nan"))
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)


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


def torch_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n):
    proj = x @ weight.T
    # The fused operator returns RMS; proj_rms_fwd returns its reciprocal.
    r = x.square().mean(-1, keepdim=True).sqrt()
    normalized = proj / (r + 1e-6)
    h_pre = (normalized[:, :n] * alpha_pre + bias[:n]).sigmoid() + 1e-6
    h_post = 2 * (normalized[:, n : 2 * n] * alpha_post + bias[n : 2 * n]).sigmoid()
    h_res = normalized[:, 2 * n :] * alpha_res + bias[2 * n :]
    return h_pre, h_post, h_res, r, proj


SHAPE_GRID = [(1, 2, 32), (7, 4, 64), (33, 2, 128)]


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", DTYPES)
def test_matches_torch(shape, dtype, weight_dtype):
    rows, streams, reduction = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = (
        torch.rand((rows, reduction), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    weight = (
        torch.rand(
            (streams * (streams + 2), reduction), device="cuda", generator=generator
        )
        * 0.2
        - 0.1
    ).to(weight_dtype)
    alphas = [torch.tensor([v], device="cuda") for v in (0.7, 0.9, 0.5)]
    bias = torch.zeros(weight.shape[0], device="cuda")
    reference_inputs = [
        v.detach().float().requires_grad_() for v in (x, weight, *alphas, bias)
    ]
    rx, rw, a_pre, a_post, a_res, rbias = reference_inputs
    expected_outputs = torch_compute_h(rx, rw, a_pre, a_post, a_res, rbias, streams)
    incoming = [
        (torch.rand(v.shape, device="cuda", generator=generator) * 0.2 - 0.1).to(
            dtype
        )
        for v in expected_outputs[:4]
    ]
    gx, gw, ga, gb, gc, gbias = torch.autograd.grad(
        expected_outputs[:4], reference_inputs, tuple(v.float() for v in incoming)
    )
    expected = (gx, gw, torch.cat((ga, gb, gc)), gbias)
    h_pre, h_post, h_res, r, proj = flash_train.torch.proj_rms_compute_h_fwd(
        x, weight, *alphas, bias, streams
    )
    actual = flash_train.torch.proj_rms_compute_h_bwd(
        *incoming, h_pre, h_post, proj, r, x, weight, *alphas, streams
    )
    rtol, atol = TOLERANCES[dtype]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        # dWeight follows weight storage, which can differ from X storage.
        storage_rtol, storage_atol = TOLERANCES[result.dtype]
        output_rtol, output_atol = max(rtol, storage_rtol), max(atol, storage_atol)
        assert torch.any(
            reference.abs() > output_atol + output_rtol * reference.abs()
        ), "reference cannot reject all-zero output"
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=output_rtol, atol=output_atol
        )
