# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


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
    q, k, w, lse = (t.squeeze(0) for t in (q, k, w, lse))
    cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    metadata = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_kv=4)

    def run(q, k, w):
        return flash_train.torch.dense_indexer_score_thd(
            q, k, w, ratio=1, sm_scale=0.2, **metadata
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


def torch_reference_dense_indexer(q, k, w, scale):
    packed = q.ndim == 3
    if packed:
        q, k, w = q[None], k[None], w[None]
    # einsum broadcasts singleton head axes; K can share one head across all Q heads.
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    scores = (logits.relu() * w.float().unsqueeze(-1)).sum(2)
    scores = scores.masked_fill(
        ~torch_reference_mask(q.shape[1], k.shape[1], q.device)[None], -torch.inf
    )
    lse = torch.logsumexp(scores, -1)
    return (scores[0], lse[0]) if packed else (scores, lse)


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["k"] = (torch.rand((a, 1, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["w"] = (torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["cu_seqlens_q"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["cu_seqlens_kv"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["q_causal_offsets"] = torch.zeros((1,), device="cuda", dtype=torch.int32)
    p["ratio"] = 1
    p["sm_scale"] = 0.25
    p["max_seqlen_kv"] = a
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])
    actual = flash_train.torch.dense_indexer_score_thd(
        q=p["q"],
        k=p["k"],
        w=p["w"],
        cu_seqlens_q=p["cu_seqlens_q"],
        cu_seqlens_kv=p["cu_seqlens_kv"],
        ratio=p["ratio"],
        sm_scale=p["sm_scale"],
        max_seqlen_kv=p["max_seqlen_kv"],
        q_causal_offsets=p["q_causal_offsets"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol, equal_nan=False
        )


@pytest.mark.parametrize("columns", [1, 63, 65, 257])
@pytest.mark.parametrize(
    "q_dtype,k_dtype",
    [
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float16),
    ],
)
def test_lse_reduction_repeated_across_waves(columns, q_dtype, k_dtype):
    # Partial waves and >256 keys exercise both shared-memory reductions.
    p = make_numerics_inputs((257, 4, 32), q_dtype)
    p["q"] = p["q"] * 10
    p["w"] = p["w"] * 10
    p["k"] = (p["k"][:columns] * 10).to(k_dtype).contiguous()
    p["cu_seqlens_kv"] = torch.tensor([0, columns], device="cuda", dtype=torch.int32)
    p["max_seqlen_kv"] = columns
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])
    rtol, atol = GRID_TOLERANCES[q_dtype] if q_dtype == k_dtype else (2e-4, 2e-5)
    for _ in range(16):
        actual = flash_train.torch.dense_indexer_score_thd(**p)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
