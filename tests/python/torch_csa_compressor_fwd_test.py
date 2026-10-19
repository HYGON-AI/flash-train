# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flash_train.torch import DSA
import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

DTYPE = torch.bfloat16


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def test_csa_compressor_matches_upstream():
    upstream = pytest.importorskip(
        "hip_dsa_kernels", reason="upstream hip_dsa_kernels is not importable"
    )
    total_tokens, head_dim, coff, ratio = 64, 32, 1, 4
    generator = torch.Generator(device="cuda").manual_seed(31)
    kv = torch.randn(
        total_tokens, coff * head_dim, device="cuda", generator=generator
    ).to(DTYPE)
    score = torch.randn(
        total_tokens, coff * head_dim, device="cuda", generator=generator
    ).to(DTYPE)
    ape = torch.randn(ratio, coff * head_dim, device="cuda", generator=generator)
    cu_seqlens = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    cu_seqlens_comp = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    total_comp = 16
    ours = flash_train.torch.csa_compressor_fwd(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp
    )
    theirs = upstream.csa_compressor_forward_wrapper(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp
    )[0]
    torch.testing.assert_close(ours, theirs)
    grad = torch.randn_like(ours)
    ours_grads = flash_train.torch.csa_compressor_bwd(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, grad, ratio, head_dim, coff
    )
    theirs_grads = upstream.csa_compressor_backward_wrapper(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, grad, ratio, head_dim, coff
    )
    for ours_grad, theirs_grad in zip(ours_grads, theirs_grads):
        torch.testing.assert_close(ours_grad, theirs_grad, rtol=1e-3, atol=1e-3)


def test_compressor_wrapper_return_contract():
    kv, score = rand(8, 32), rand(8, 32)
    ape = rand(4, 32, dtype=torch.float32)
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    comp = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    out = DSA.csa_compressor_forward_wrapper(kv, score, ape, cu, comp, 4, 32, 1, 2)[
        "out"
    ]
    grad = torch.ones_like(out)
    grads = DSA.csa_compressor_backward_wrapper(
        kv, score, ape, cu, comp, grad, 4, 32, 1
    )
    assert isinstance(grads, tuple) and len(grads) == 3
    gkv, gs, gape = grads
    for got, expected in zip(
        (gkv, gs, gape), ft.csa_compressor_bwd(kv, score, ape, cu, comp, grad, 4, 32, 1)
    ):
        torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("total_comp", [0, 16])
def test_rejects_unsupported_compute_including_empty_output(dtype, total_comp):
    kv = torch.zeros(64, 32, device="cuda", dtype=dtype)
    score = torch.zeros_like(kv)
    ape = torch.zeros(4, 32, device="cuda")
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    comp = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="failed with status"):
        ft.csa_compressor_fwd(kv, score, ape, cu, comp, 4, 32, 1, total_comp)


def test_rejects_zero_coff_before_dimension_division():
    kv = torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16)
    ape = torch.zeros(4, 32, device="cuda")
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    comp = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="failed with status 1"):
        ft.csa_compressor_fwd(kv, kv, ape, cu, comp, 4, 32, 0, 16)


# Non-power-of-two boundaries and every supported storage precision.
SHAPE_GRID = [(8, 4, 32), (36, 4, 64), (128, 4, 128)]
GRID_DTYPES = [torch.bfloat16]
# Tolerances cover reduction error and FP16/BF16 storage rounding for
# inputs bounded to [-0.1, 0.1]; FP32 outputs retain the stricter tolerance.
GRID_TOLERANCES = {
    torch.bfloat16: (0.05, 0.005),
}


def torch_reference_compressor(kv, score, ape, ratio):
    groups = kv.shape[0] // ratio
    keys = kv[: groups * ratio].float().reshape(groups, ratio, -1)
    logits = (
        score[: groups * ratio].float().reshape(groups, ratio, -1) + ape.float()[None]
    )
    return (keys * logits.softmax(1)).sum(1)


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
    p["ratio"] = b
    p["head_dim"] = c
    p["coff"] = 1
    p["total_comp"] = a // b
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_compressor(p["kv"], p["score"], p["ape"], p["ratio"]).to(
        p["kv"].dtype
    )
    actual = flash_train.torch.csa_compressor_fwd(
        kv=p["kv"],
        score=p["score"],
        ape=p["ape"],
        cu_seqlens=p["cu_seqlens"],
        cu_seqlens_comp=p["cu_seqlens_comp"],
        ratio=p["ratio"],
        head_dim=p["head_dim"],
        coff=p["coff"],
        total_comp=p["total_comp"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=rtol, atol=atol, equal_nan=False
    )
