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

SHAPE_GRID = [
    (1, 2, 16),
    (7, 4, 32),
    (33, 4, 64),
    (3, 8, 17),
    (257, 4, 40),
    (3, 16, 32),
]


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_torch(shape, dtype):
    rows, streams, channels = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    residual = (
        torch.rand((rows, 1, streams, channels), device="cuda", generator=generator)
        * 0.2
        - 0.1
    ).to(dtype)
    x = (
        torch.rand((rows, 1, channels), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    h_res = (
        torch.rand((rows, 1, streams, streams), device="cuda", generator=generator)
        * 0.2
        - 0.1
    ).to(dtype)
    h_post = (
        torch.rand((rows, 1, streams), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    bias = torch.rand((channels,), device="cuda", generator=generator) * 0.2 - 0.1
    grad = (
        torch.rand(residual.shape, device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    reference_inputs = [
        v.detach().float().requires_grad_() for v in (residual, x, h_res, h_post, bias)
    ]
    rr, rx, rh_res, rh_post, rbias = reference_inputs
    expected_output = torch.einsum(
        "...ij,...ic->...jc", rh_res, rr
    ) + rh_post.unsqueeze(-1) * (rx + rbias).unsqueeze(-2)
    expected = torch.autograd.grad(expected_output, reference_inputs, grad.float())
    actual = flash_train.torch.h_post_bda_bwd(grad, residual, x, h_res, h_post, bias)
    rtol, atol = TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_present", [False, True])
@pytest.mark.parametrize("want_bias_gradient", [False, True])
def test_optional_bias_and_requested_bias_gradient(
    dtype, bias_present, want_bias_gradient
):
    residual = torch.ones(3, 1, 2, 16, device="cuda", dtype=dtype)
    x = torch.ones(3, 1, 16, device="cuda", dtype=dtype)
    h_res = torch.ones(3, 1, 2, 2, device="cuda", dtype=dtype)
    h_post = torch.ones(3, 1, 2, device="cuda", dtype=dtype)
    bias = torch.zeros(16, device="cuda") if bias_present else None
    actual = flash_train.torch.h_post_bda_bwd(
        torch.ones_like(residual),
        residual,
        x,
        h_res,
        h_post,
        bias,
        want_bias_gradient=want_bias_gradient,
    )
    if want_bias_gradient:
        # dBias = sum_{S,B,N}(grad_output * h_post), independent of the bias value.
        torch.testing.assert_close(actual[-1], torch.full((16,), 6.0, device="cuda"))
    else:
        assert actual[-1] is None
    torch.testing.assert_close(actual[0], torch.full_like(residual, 2))
    torch.testing.assert_close(actual[1], torch.full_like(x, 2))
    torch.testing.assert_close(actual[2], torch.full_like(h_res, 16))
    torch.testing.assert_close(actual[3], torch.full_like(h_post, 16))
