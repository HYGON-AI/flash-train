# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_packed_indexer_and_scores_match_upstream(dtype):
    upstream = pytest.importorskip(
        "hip_dsa_kernels", reason="upstream hip_dsa_kernels is not importable"
    )
    generator = torch.Generator(device="cuda").manual_seed(79)
    q = torch.randn(8, 4, 32, device="cuda", generator=generator).to(dtype)
    k = torch.randn(8, 1, 32, device="cuda", generator=generator).to(dtype)
    w = torch.randn(8, 4, device="cuda", generator=generator).to(dtype)
    lse = torch.randn(8, 4, device="cuda", generator=generator)
    cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    meta = dict(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_kv=4, q_causal_offsets=offsets
    )
    scores = flash_train.torch.indexer_fwd_thd(q, k, w, ratio=1, sm_scale=0.2, **meta)
    theirs = upstream.indexer_fwd(q, k, w, 1, 0.2, cu, cu, 4, 4, offsets)
    torch.testing.assert_close(scores, theirs)
    index_score = flash_train.torch.dense_indexer_score_thd(
        q, k, w, ratio=1, sm_scale=0.2, **meta
    )
    theirs_index = upstream.dense_indexer_score(q, k, w, 1, 0.2, cu, cu, 4, 4, offsets)
    target = flash_train.torch.dense_attn_score_thd(
        q, k, lse, ratio=1, softmax_scale=0.2, **meta
    )
    theirs_target = upstream.dense_attn_score(q, k, lse, 0.2, 1, cu, cu, 4, 4, offsets)
    for got, ref in zip((*index_score, *target), (*theirs_index, *theirs_target)):
        torch.testing.assert_close(got, ref)
    got_topk = flash_train.torch.indexer_topk_thd(
        scores, cu, cu, 1, 2, order_values=True, q_causal_offsets=offsets
    )
    ref_topk = upstream.indexer_topk_thd(theirs, cu, cu, 1, 2, True, offsets)
    for got, ref in zip(got_topk, ref_topk):
        torch.testing.assert_close(got, ref)
    got_grad = flash_train.torch.indexer_bwd_dense_thd(
        q,
        w,
        k.squeeze(1),
        *target,
        index_score[0].clone(),
        index_score[1],
        cu,
        cu,
        sm_scale=0.2,
        ratio=1,
        q_causal_offsets=offsets,
    )
    ref_grad = upstream.indexer_bwd_dense(
        q,
        w,
        k.squeeze(1),
        *theirs_target,
        theirs_index[0].clone(),
        theirs_index[1],
        0.2,
        1.0,
        1.0,
        1,
        cu,
        cu,
        4,
        4,
        offsets,
    )
    for got, ref in zip(got_grad[1:], ref_grad):
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


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
        return flash_train.torch.indexer_fwd_thd(
            q, k, w, ratio=1, sm_scale=0.2, **metadata
        )

    expected = run(q.float(), k.float(), w.float())
    # Repeat the same signature so both cold selection and cache reuse execute.
    for _ in range(2):
        actual = run(q, k, w)
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-5, equal_nan=True
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
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])[0]
    actual = flash_train.torch.indexer_fwd_thd(
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
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=rtol, atol=atol, equal_nan=True
    )
