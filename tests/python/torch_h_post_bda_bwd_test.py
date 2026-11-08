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
@pytest.mark.parametrize("mapping_fp32", [False, True])
def test_matches_torch(shape, dtype, mapping_fp32):
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
    if mapping_fp32:
        h_res, h_post = h_res.float(), h_post.float()
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
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert torch.any(
            reference.abs() > atol + rtol * reference.abs()
        ), "reference cannot reject all-zero output"
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mapping_fp32", [False, True])
@pytest.mark.parametrize("bias_dtype", [None, torch.float16, torch.float32])
def test_autograd_wrapper_optional_bias(dtype, mapping_fp32, bias_dtype):
    torch.manual_seed(823)
    mapping_dtype = torch.float32 if mapping_fp32 else dtype
    shapes = [(7, 1, 4, 4), (7, 1, 4, 32), (7, 1, 4), (7, 1, 32)]
    dtypes = [mapping_dtype, dtype, mapping_dtype, dtype]
    inputs = [
        (torch.randn(shape, device="cuda") * 0.1).to(dt).requires_grad_()
        for shape, dt in zip(shapes, dtypes)
    ]
    bias = None
    if bias_dtype is not None:
        bias = (torch.randn(32, device="cuda") * 0.1).to(bias_dtype).requires_grad_()
        inputs.append(bias)
    refs = [v.detach().float().requires_grad_() for v in inputs]
    h_res, residual, h_post, x = refs[:4]
    expected = torch.einsum("...ij,...ic->...jc", h_res, residual)
    expected = expected + h_post.unsqueeze(-1) * (
        x if bias is None else x + refs[4]
    ).unsqueeze(-2)
    actual = flash_train.torch.fused_h_post_bda(*inputs[:4], bias)
    # Exercise the wrapper's contiguous conversion for incoming gradients.
    grad = torch.randn(7, 1, 32, 4, device="cuda", dtype=dtype).transpose(-1, -2)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, refs, grad.float())
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
    for value, reference, operand in zip(actual_grads, expected_grads, inputs):
        assert value.dtype == operand.dtype
        torch.testing.assert_close(value, reference.to(value.dtype), rtol=rtol, atol=atol)
