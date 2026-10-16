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
def test_sparse_attn_target_matches_upstream(k_dtype):
    upstream = pytest.importorskip(
        "hip_dsa_kernels", reason="upstream hip_dsa_kernels is not importable"
    )
    q, k, _ = make_indexer()
    k = k.to(k_dtype)
    lse = torch.randn(q.shape[0], q.shape[1], q.shape[2], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(3)
    topk = torch.randint(
        0,
        k.shape[1],
        (q.shape[0], q.shape[1], 6),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    )
    ours = flash_train.torch.sparse_attn_target(q, k, lse, topk, softmax_scale=0.5)
    theirs = upstream.sparse_attn_target(q, k, lse, topk, 0.5, False)
    torch.testing.assert_close(ours, theirs, equal_nan=True)


@pytest.mark.parametrize("global_indices", [False, True])
def test_score_recompute_prefix_against_torch(global_indices):
    torch.manual_seed(527)
    b, t, h, s, d, k = 2, 6, 32, 12, 576, 8
    q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(b, s, d, device="cuda", dtype=torch.bfloat16) * 0.1
    lse = torch.randn(b, t, h, device="cuda") * 0.1
    idx = torch.randint(s, (b, t, k), device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [[-2, 0, 1, 4, 8, 11], [2, 3, 5, 6, 7, 8]], device="cuda", dtype=torch.int32
    )
    if global_indices:
        idx[1] += s
    before = idx.clone()
    want = torch.zeros(b, t, k, device="cuda")
    for batch in range(b):
        for row in range(t):
            n = max(0, min(k, int(lengths[batch, row])))
            if not n:
                continue
            keys = idx[batch, row, :n].long() - (batch * s if global_indices else 0)
            logits = q[batch, row].float() @ kv[batch, keys].float().T * 0.04
            scores = (logits - lse[batch, row, :, None]).exp().sum(0)
            want[batch, row, :n] = scores / scores.sum()
    args = (q, kv, lse, idx, 0.04)
    kwargs = dict(
        topk_length=lengths, qhead_per_kv_head=h, topk_indices_global=global_indices
    )
    got = DSA.sparse_attn_score_recompute_wrapper(*args, **kwargs)["target"]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)
    assert torch.equal(idx, before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = DSA.sparse_attn_score_recompute_wrapper(*args, **kwargs)["target"]
    lengths.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(captured) == 0


def test_sparse_recompute_positional_scale_and_keyword_arguments():
    q, k, w = rand(2, 4, 4, 32), rand(2, 8, 32), rand(2, 4, 4)
    topk = torch.randint(0, 8, (2, 4, 4), device="cuda", dtype=torch.int32)
    topk[1] += 8
    lse = rand(2, 4, 4, dtype=torch.float32)
    predicted = DSA.sparse_indexer_score_recompute_wrapper(
        q=q, k=k, w=w, topk_idxs=topk, sm_scale=0.2, topk_indices_global=True
    )["predict"]
    target = DSA.sparse_attn_score_recompute_wrapper(
        q, k, lse, topk, 0.2, topk_indices_global=True
    )["target"]
    torch.testing.assert_close(
        predicted,
        ft.sparse_indexer_predict(
            q, k, w, topk, sm_scale=0.2, topk_indices_global=True
        ),
    )
    got = DSA.indexer_backward_wrapper(
        index_q=q,
        weights=w,
        index_k=k,
        attn_score=target.clone(),
        index_score=predicted,
        topk_indices=topk,
        sm_scale=0.2,
        topk_indices_global=True,
        block_I=64,
    )
    expected = ft.indexer_bwd_sparse(
        q, w, k, target.clone(), predicted, topk, sm_scale=0.2, topk_indices_global=True
    )
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
    indices = torch.randint(8, (1, 8, 4), device="cuda", dtype=torch.int32)
    k = k.squeeze(-2)

    def run(q, k, w):
        return flash_train.torch.sparse_attn_target(
            q, k, lse, indices, softmax_scale=0.2
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


@pytest.mark.parametrize("heads,padded", [(32, False), (64, False), (64, True)])
def test_d576_target_fast_path(heads, padded):
    torch.manual_seed(610)
    q = rand(1, 4096, heads, 576) * 0.1
    kv = rand(1, 4096, 576) * 0.1
    lse = torch.randn(1, 4096, heads, device="cuda") * 0.1
    if padded:
        q[:, :, 32:] = 0
        lse[:, :, 32:] = float("inf")
    indices = torch.full((1, 4096, 2048), -1, device="cuda", dtype=torch.int32)
    indices[:, :, :128] = torch.arange(128, device="cuda", dtype=torch.int32)
    scale = 576**-0.5
    result = ft.sparse_attn_target(q, kv, lse, indices, softmax_scale=scale)
    rows = torch.tensor([0, 2047, 4095], device="cuda")
    score = torch.einsum("thd,kd->thk", q[0, rows].float(), kv[0, :128].float()) * scale
    raw = (score - lse[0, rows, :, None]).exp().sum(1)
    expected = raw / raw.sum(-1, keepdim=True)
    torch.testing.assert_close(result[0, rows, :128], expected, rtol=2e-2, atol=2e-4)
    assert torch.count_nonzero(result[:, :, 128:]) == 0


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


def torch_reference_gathered(q, k, indices, scale):
    # Index a 2-D matrix per batch: this HCU Torch build mis-shapes a
    # multidimensional fancy index against a batched 3-D matrix.
    keys = torch.stack([k[b].float()[indices[b].long()] for b in range(q.shape[0])])
    logits = (q.float().unsqueeze(3) * keys.unsqueeze(2)).sum(-1) * scale
    return keys, logits


def torch_reference_sparse_target(q, k, lse, indices, scale):
    _, logits = torch_reference_gathered(q, k, indices, scale)
    target = (logits - lse.float().unsqueeze(-1)).exp().sum(2)
    return target / target.sum(-1, keepdim=True)


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (
        torch.rand((1, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["k"] = (torch.rand((1, a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["lse"] = (
        torch.rand((1, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["topk_indices"] = (
        (torch.arange(min(a, 32), device="cuda", dtype=torch.int32) % a)
        .expand((1, a, min(a, 32)))
        .contiguous()
    )
    p["softmax_scale"] = 0.125
    p["topk_indices_global"] = 0
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_sparse_target(
        p["q"], p["k"], p["lse"], p["topk_indices"], p["softmax_scale"]
    )
    actual = flash_train.torch.sparse_attn_target(
        q=p["q"],
        k=p["k"],
        lse=p["lse"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
        topk_indices_global=p["topk_indices_global"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=rtol, atol=atol, equal_nan=True
    )
