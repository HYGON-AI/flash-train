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


def make_indexer(batch=2, seq_q=8, seq_k=16, heads=4, head_dim=64, seed=17):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch, seq_q, heads, head_dim, device="cuda", generator=generator
    ).to(DTYPE)
    k = torch.randn(batch, seq_k, head_dim, device="cuda", generator=generator).to(
        DTYPE
    )
    w = torch.randn(batch, seq_q, heads, device="cuda", generator=generator).to(DTYPE)
    return q, k, w


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.parametrize("k_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_dense_attn_score_matches_upstream(k_dtype):
    upstream = pytest.importorskip(
        "hip_dsa_kernels", reason="upstream hip_dsa_kernels is not importable"
    )
    q, k, _ = make_indexer()
    k = k.to(k_dtype)
    lse = torch.randn(q.shape[0], q.shape[1], q.shape[2], device="cuda")
    ours_out, ours_denom = flash_train.torch.dense_attn_score(
        q, k.unsqueeze(2), lse, ratio=4, softmax_scale=0.5
    )
    theirs_out, theirs_denom = upstream.dense_attn_score(
        q, k.unsqueeze(2), lse, 0.5, 4, None, None, 0, 0, None
    )
    torch.testing.assert_close(ours_out, theirs_out, equal_nan=False)
    torch.testing.assert_close(ours_denom, theirs_denom)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
def test_dense_recompute_framework_signatures(dtype, packed):
    torch.manual_seed(71)
    q, k, w = (
        rand(1, 8, 4, 32, dtype=dtype),
        rand(1, 8, 1, 32, dtype=dtype),
        rand(1, 8, 4, dtype=dtype),
    )
    lse = rand(1, 8, 4, dtype=torch.float32)
    metadata = {}
    if packed:
        q, k, w, lse = (t.squeeze(0) for t in (q, k, w, lse))
        cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        metadata = dict(
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_k=4,
            max_seqlen_q=4,
            q_causal_offsets=offsets,
        )
        pure = dict(
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_kv=4, q_causal_offsets=offsets
        )
        expected_i = ft.dense_indexer_score_thd(q, k, w, ratio=1, sm_scale=0.2, **pure)
        expected_a = ft.dense_attn_score_thd(
            q, k, lse, ratio=1, softmax_scale=0.2, **pure
        )
    else:
        expected_i = ft.dense_indexer_score(q, k, w, ratio=1, sm_scale=0.2)
        expected_a = ft.dense_attn_score(q, k, lse, ratio=1, softmax_scale=0.2)
    got_i = DSA.dense_indexer_score_recompute_wrapper(
        q=q, k=k, w=w, sm_scale=0.2, qhead_per_kv_head=4, **metadata
    )
    got_a = DSA.dense_attn_score_recompute_wrapper(
        q, k, lse, 0.2, qhead_per_kv_head=4, **metadata
    )
    for got, expected in ((got_i, expected_i), (got_a, expected_a)):
        for name, value in zip(("out", "denom"), expected):
            torch.testing.assert_close(got[name], value)
    kwargs = dict(
        index_q=q,
        weights=w,
        index_k=k.squeeze(-2),
        attn_score=got_a["out"],
        attn_l1norm=got_a["denom"],
        index_score=got_i["out"].clone(),
        index_lse=got_i["denom"],
        sm_scale=0.2,
    )
    # The default ratio must be one, and keyword tensor arguments must survive.
    got = DSA.dense_indexer_backward_wrapper(**kwargs, **metadata, block_I=64)
    kwargs["index_score"] = expected_i[0].clone()
    if packed:
        expected = ft.indexer_bwd_dense_thd(
            **kwargs,
            ratio=1,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            q_causal_offsets=offsets,
        )
    else:
        expected = ft.indexer_bwd_dense(**kwargs, ratio=1)
    for name, value in zip(("d_index_q", "d_weights", "d_index_k"), expected[1:]):
        torch.testing.assert_close(got[name], value, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "types",
    [
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float16),
    ],
)
def test_mixed_storage_matches_fp32(types):
    torch.manual_seed(615)
    q = torch.randn(1, 8, 4, 32, device="cuda").to(types[0])
    k = torch.randn(1, 8, 1, 32, device="cuda").to(types[1])
    w = torch.randn(1, 8, 4, device="cuda").to(types[2])
    lse = torch.randn(1, 8, 4, device="cuda")
    metadata = {}

    def run(q, k, w):
        return flash_train.torch.dense_attn_score(
            q, k, lse, ratio=1, softmax_scale=0.2, **metadata
        )

    expected = run(q.float(), k.float(), w.float())
    # Repeat the same signature so both cold selection and cache reuse execute.
    for _ in range(2):
        actual = run(q, k, w)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(
                value, reference, rtol=2e-4, atol=2e-5, equal_nan=False
            )
    with pytest.raises(RuntimeError):
        run(q, k.to(torch.int32), w)


# Non-power-of-two boundaries and every supported storage precision.
SHAPE_GRID = [(8, 2, 32), (33, 2, 64), (128, 4, 128)]
GRID_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Tolerances cover reduction error and FP16/BF16 storage rounding for
# inputs bounded to [-0.1, 0.1]; FP32 outputs retain the stricter tolerance.
GRID_TOLERANCES = {
    torch.float32: (0.0002, 2e-05),
    torch.float16: (0.02, 0.002),
    torch.bfloat16: (0.05, 0.005),
}


def torch_reference_mask(rows, columns, device):
    return (
        torch.arange(columns, device=device)[None, :]
        <= torch.arange(rows, device=device)[:, None]
    )


def torch_reference_dense_attn(q, k, lse, scale):
    packed = q.ndim == 3
    if packed:
        q, k, lse = q[None], k[None], lse[None]
    # einsum broadcasts singleton head axes; K can share one head across all Q heads.
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    values = (logits - lse.float().unsqueeze(-1)).exp().sum(2)
    valid = torch_reference_mask(q.shape[1], k.shape[1], q.device)[None]
    denom = values.masked_fill(~valid, 0).sum(-1)
    values = values.masked_fill(~valid, -torch.inf)
    return (values[0], denom[0]) if packed else (values, denom)


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (
        torch.rand((1, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["k"] = (
        torch.rand((1, a, 1, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["lse"] = (
        torch.rand((1, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["ratio"] = 1
    p["softmax_scale"] = 0.125
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_dense_attn(p["q"], p["k"], p["lse"], p["softmax_scale"])
    actual = flash_train.torch.dense_attn_score(
        q=p["q"],
        k=p["k"],
        lse=p["lse"],
        ratio=p["ratio"],
        softmax_scale=p["softmax_scale"],
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
