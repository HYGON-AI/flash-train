# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

DTYPE = torch.bfloat16


import flash_train.torch as ft


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two devices")
def test_current_device_and_cross_device_rejection():
    previous = torch.cuda.current_device()
    other = (previous + 1) % torch.cuda.device_count()
    x = torch.ones(2, 1, 2, 16, device=f"cuda:{other}", dtype=torch.bfloat16)
    h = torch.ones(2, 1, 2, device=f"cuda:{other}")
    with torch.cuda.device(other):
        out = ft.h_aggregate_fwd(x, h)
        torch.testing.assert_close(out, torch.full_like(out, 2))
    assert torch.cuda.current_device() == previous
    with pytest.raises(RuntimeError, match="same device"):
        ft.h_aggregate_fwd(x, h.to(f"cuda:{previous}"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_empty_batch_forward_and_backward_are_noops(dtype):
    x = torch.empty(0, 1, 4, 16, device="cuda", dtype=dtype)
    h_pre = torch.empty(0, 1, 4, device="cuda")
    actual = flash_train.torch.h_aggregate_fwd(x, h_pre)
    gx, gh = flash_train.torch.h_aggregate_bwd(torch.empty_like(actual), x, h_pre)
    torch.cuda.synchronize()
    assert actual.shape == (0, 1, 16)
    assert gx.shape == x.shape
    assert gh.shape == h_pre.shape


def test_zero_channels_forward_is_noop_and_backward_is_unsupported():
    x = torch.empty(2, 1, 4, 0, device="cuda")
    h_pre = torch.ones(2, 1, 4, device="cuda")
    output = flash_train.torch.h_aggregate_fwd(x, h_pre)
    torch.cuda.synchronize()
    assert output.shape == (2, 1, 0)
    with pytest.raises(RuntimeError, match="failed with status"):
        flash_train.torch.h_aggregate_bwd(output, x, h_pre)


@pytest.mark.parametrize(
    "dtype, expected", [(torch.bfloat16, 2.5), (torch.float16, 4.6875)]
)
def test_equal_width_float_formats_are_not_reinterpreted(dtype, expected):
    # 0x3f80 means 1.0 in BF16 and 1.875 in FP16. Multiplication must use
    # the selected numeric format, despite identical storage width and bits.
    bits = torch.full((1, 1, 1, 1), 0x3F80, device="cuda", dtype=torch.int16)
    x = bits.view(dtype)
    h_pre = torch.full((1, 1, 1), 2.5, device="cuda", dtype=torch.float32)
    actual = flash_train.torch.h_aggregate_fwd(x, h_pre)
    torch.testing.assert_close(
        actual, torch.full_like(actual, expected), rtol=0, atol=0
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
    expected = (x.float() * h.unsqueeze(-1)).sum(-2).to(dtype)
    actual = flash_train.torch.h_aggregate_fwd(x, h)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
