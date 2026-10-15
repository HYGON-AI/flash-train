# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

DTYPE = torch.bfloat16


def test_empty_forward_is_noop_and_backward_rejects_nonempty_bias_gradient():
    residual = torch.empty(0, 1, 4, 16, device="cuda")
    h_res = torch.empty(0, 1, 4, 4, device="cuda")
    h_post = torch.empty(0, 1, 4, device="cuda")
    x = torch.empty(0, 1, 16, device="cuda")
    bias = torch.ones(16, device="cuda")
    output = flash_train.torch.h_post_bda_fwd(h_res, residual, h_post, x, bias)
    torch.cuda.synchronize()
    assert output.shape == residual.shape
    with pytest.raises(RuntimeError, match="failed with status"):
        flash_train.torch.h_post_bda_bwd(output, residual, x, h_res, h_post, bias)


def test_zero_channels_forward_is_noop_and_backward_is_unsupported():
    residual = torch.empty(2, 1, 4, 0, device="cuda")
    h_res = torch.ones(2, 1, 4, 4, device="cuda")
    h_post = torch.ones(2, 1, 4, device="cuda")
    x = torch.empty(2, 1, 0, device="cuda")
    output = flash_train.torch.h_post_bda_fwd(h_res, residual, h_post, x)
    torch.cuda.synchronize()
    assert output.shape == residual.shape
    with pytest.raises(RuntimeError, match="failed with status"):
        flash_train.torch.h_post_bda_bwd(output, residual, x, h_res, h_post)


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
    expected = (
        torch.einsum("...ij,...ic->...jc", h_res.float(), residual.float())
        + h_post.float().unsqueeze(-1) * (x.float() + bias).unsqueeze(-2)
    ).to(dtype)
    actual = flash_train.torch.h_post_bda_fwd(h_res, residual, h_post, x, bias)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
