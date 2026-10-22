# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("total_comp", [0, 16])
def test_rejects_unsupported_compute_including_empty_gradient(dtype, total_comp):
    kv = torch.zeros(64, 32, device="cuda", dtype=dtype)
    score = torch.zeros_like(kv)
    ape = torch.zeros(4, 32, device="cuda")
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    comp = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    grad = torch.zeros(total_comp, 32, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="failed with status"):
        ft.csa_compressor_bwd(kv, score, ape, cu, comp, grad, 4, 32, 1)


def test_rejects_zero_coff_before_dimension_division():
    kv = torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16)
    ape = torch.zeros(4, 32, device="cuda")
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    comp = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    grad = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="failed with status 1"):
        ft.csa_compressor_bwd(kv, kv, ape, cu, comp, grad, 4, 32, 0)


# Non-power-of-two boundaries and every supported storage precision.
SHAPE_GRID = [(8, 4, 32), (36, 4, 64), (128, 4, 128)]
GRID_DTYPES = [torch.bfloat16]
# Relative tolerance covers BF16 gradient storage rounding; the absolute budget
# stays below the small score/APE gradients for inputs bounded to [-0.1, 0.1].
GRID_TOLERANCES = {
    torch.bfloat16: (0.05, 2e-05),
}


def torch_reference_compressor(kv, score, ape, ratio):
    groups = kv.shape[0] // ratio
    keys = kv[: groups * ratio].float().reshape(groups, ratio, -1)
    logits = (
        score[: groups * ratio].float().reshape(groups, ratio, -1) + ape.float()[None]
    )
    return (keys * logits.softmax(1)).sum(1)


def torch_reference_compressor_backward(p):
    names = ("kv", "score", "ape")
    values = [p[n].detach().float().requires_grad_() for n in names]
    output = torch_reference_compressor(*values, p["ratio"])
    grads = torch.autograd.grad(output, values, p["grad_compressed"].float())
    return tuple(t.to(p[n].dtype) for t, n in zip(grads, names))


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["kv"] = (torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.bfloat16
    )
    p["score"] = (
        torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.bfloat16)
    p["ape"] = (torch.rand((b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.float32
    )
    p["cu_seqlens"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["cu_seqlens_comp"] = torch.tensor([0, a // b], device="cuda", dtype=torch.int32)
    p["grad_compressed"] = (
        torch.rand((a // b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.bfloat16)
    p["ratio"] = b
    p["head_dim"] = c
    p["coff"] = 1
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_compressor_backward(p)
    actual = flash_train.torch.csa_compressor_bwd(
        kv=p["kv"],
        score=p["score"],
        ape=p["ape"],
        cu_seqlens=p["cu_seqlens"],
        cu_seqlens_comp=p["cu_seqlens_comp"],
        grad_compressed=p["grad_compressed"],
        ratio=p["ratio"],
        head_dim=p["head_dim"],
        coff=p["coff"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert torch.any(
            reference.abs() > atol + rtol * reference.abs()
        ), "reference cannot reject all-zero output"
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol, equal_nan=False
        )
