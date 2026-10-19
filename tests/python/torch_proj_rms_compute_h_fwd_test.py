# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("n", [2, 4])
def test_proj_rms_compute_h_matches_upstream(dtype, weight_dtype, n):
    upstream = pytest.importorskip(
        "fused_mhc_ops", reason="upstream fused_mhc_ops is not importable"
    )
    generator = torch.Generator(device="cuda").manual_seed(51)
    outputs = n * n + 2 * n
    x = torch.randn(64, 32, device="cuda", generator=generator).to(dtype)
    weight = torch.randn(outputs, 32, device="cuda", generator=generator).to(
        weight_dtype
    )
    alpha_pre = torch.randn(1, device="cuda", generator=generator)
    alpha_post = torch.randn(1, device="cuda", generator=generator)
    alpha_res = torch.randn(1, device="cuda", generator=generator)
    bias = torch.randn(outputs, device="cuda", generator=generator)

    theirs = upstream.proj_rms_compute_h_forward(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    ours = flash_train.torch.proj_rms_compute_h_fwd(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    for index in range(4):
        torch.testing.assert_close(ours[index], theirs[index], rtol=1e-2, atol=1e-2)

    h_pre, h_post, h_res, r, _proj = ours
    grad_h_pre = torch.randn_like(h_pre, generator=generator).to(dtype)
    grad_h_post = torch.randn_like(h_post, generator=generator).to(dtype)
    grad_h_res = torch.randn_like(h_res, generator=generator).to(dtype)
    grad_r = torch.randn_like(r, generator=generator).to(dtype)
    theirs_grads = upstream.proj_rms_compute_h_backward(
        grad_h_pre,
        grad_h_post,
        grad_h_res,
        grad_r,
        h_pre,
        h_post,
        theirs[4],
        r,
        x,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        n,
        1e-6,
        1e-6,
    )
    ours_grads = flash_train.torch.proj_rms_compute_h_bwd(
        grad_h_pre,
        grad_h_post,
        grad_h_res,
        grad_r,
        h_pre,
        h_post,
        ours[4],
        r,
        x,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        n,
        1e-6,
        1e-6,
    )
    for ours_grad, theirs_grad in zip(ours_grads, theirs_grads):
        torch.testing.assert_close(ours_grad, theirs_grad, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_split_k_forward_workspace_and_graph_match_upstream(weight_dtype):
    upstream = pytest.importorskip(
        "fused_mhc_ops", reason="upstream fused_mhc_ops is not importable"
    )
    # Both forward families split K for this grid and must use Plan workspace.
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(8, 4096, device="cuda", dtype=weight_dtype) * 0.1
    alphas = [torch.ones(1, device="cuda") for _ in range(3)]
    bias = torch.zeros(8, device="cuda")
    expected_plain = upstream.proj_rms_forward(x, weight, 1e-6)
    expected_h = upstream.proj_rms_compute_h_forward(
        x, weight, *alphas, bias, 2, 1e-6, 1e-6
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flash_train.torch.proj_rms_fwd(x, weight)
        flash_train.torch.proj_rms_compute_h_fwd(x, weight, *alphas, bias, 2)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual_plain = flash_train.torch.proj_rms_fwd(x, weight)
        actual_h = flash_train.torch.proj_rms_compute_h_fwd(x, weight, *alphas, bias, 2)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(
            (*actual_plain, *actual_h), (*expected_plain, *expected_h)
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("bias_present", [False, True])
def test_mhc_autograd_chain_with_fp32_projection_weights(bias_present):
    torch.manual_seed(73)
    residual = (rand(4, 2, 2, 16) * 0.1).requires_grad_()
    weight = (rand(8, 32, dtype=torch.float32) * 0.1).requires_grad_()
    alphas = [
        torch.ones(1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    bias_h = torch.zeros(8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hp, hpost, hr, r = ft.mhc_wrappers.fused_proj_rms_compute_h(
        residual.reshape(8, 32), weight, *alphas, bias_h, 2
    )
    hp, hpost = hp.reshape(4, 2, 2), hpost.reshape(4, 2, 2)
    hr = ft.mhc_wrappers.fused_sinkhorn(hr.reshape(4, 2, 2, 2), 3)
    aggregated = ft.mhc_wrappers.fused_h_aggregate(residual, hp)
    bias = (
        torch.zeros(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if bias_present
        else None
    )
    out = ft.mhc_wrappers.fused_h_post_bda(hr, residual, hpost, aggregated, bias)
    out.float().square().mean().backward()
    for value in [residual, weight, bias_h, *alphas] + ([bias] if bias_present else []):
        assert value.grad is not None
        assert value.grad.dtype == value.dtype
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("ksplit", [2, 3, 4, 64])
def test_proj_rms_compute_h_ksplit_override(ksplit):
    # MHX_KS is captured once by each kernel implementation.
    script = """
import os
import torch
import flash_train.torch

split = int(os.environ["MHX_KS"])
for dtype, weight_dtype in [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.float32),
]:
    x = torch.ones(4, 32, device="cuda", dtype=dtype)
    weight = torch.ones(24, 32, device="cuda", dtype=weight_dtype)
    alpha = torch.ones(1, device="cuda")
    bias = torch.zeros(24, device="cuda")
    fragment = 16 if dtype == weight_dtype and dtype != torch.float32 else 8
    valid = 32 % split == 0 and (32 // split) % fragment == 0
    try:
        *_, r, projection = flash_train.torch.proj_rms_compute_h_fwd(
            x, weight, alpha, alpha, alpha, bias, 4
        )
    except RuntimeError as error:
        assert not valid, str(error)
        assert "failed with status 1" in str(error), str(error)
    else:
        assert valid, "invalid split was accepted"
        torch.testing.assert_close(projection.float(), torch.full_like(projection.float(), 32))
        torch.testing.assert_close(r.float(), torch.ones_like(r.float()), rtol=1e-2, atol=1e-2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "MHX_KS": str(ksplit)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    expected = torch_compute_h(x.float(), weight.float(), *alphas, bias, streams)
    actual = flash_train.torch.proj_rms_compute_h_fwd(x, weight, *alphas, bias, streams)
    rtol, atol = TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_split_k_matches_torch_without_upstream(dtype, weight_dtype):
    # Small grids and long reductions exercise cross-wave partial publication in split-K.
    torch.manual_seed(720)
    rows, n, width = 65, 4, 4096
    x = (torch.randn(rows, width, device="cuda") * 0.1).to(dtype)
    weight = (torch.randn(n * n + 2 * n, width, device="cuda") * 0.1).to(weight_dtype)
    alphas = [torch.tensor([v], device="cuda") for v in (0.5, -0.25, 0.75)]
    bias = torch.randn(n * n + 2 * n, device="cuda") * 0.1
    ft.proj_rms_compute_h_fwd(x, weight, *alphas, bias, n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ft.proj_rms_compute_h_fwd(x, weight, *alphas, bias, n)
    for repeat in range(8):
        x.mul_(-0.875)
        proj = x.float() @ weight.float().T
        r = x.float().square().mean(-1, keepdim=True).sqrt()
        normalized = proj / (r + 1e-6)
        expected = (
            (normalized[:, :n] * alphas[0] + bias[:n]).sigmoid() + 1e-6,
            2 * (normalized[:, n : 2 * n] * alphas[1] + bias[n : 2 * n]).sigmoid(),
            normalized[:, 2 * n :] * alphas[2] + bias[2 * n :],
            r,
            proj,
        )
        graph.replay()
        rtol, atol = (2e-4, 2e-5) if dtype == torch.float32 else (5e-2, 5e-3)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference.to(dtype), rtol=rtol, atol=atol)
