# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


def ref_indexer_fwd(q, k, w, ratio, sm_scale):
    """scores[b,q,k] = mask * sum_h ReLU(q.k^T * sm_scale) * w."""
    scores = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * sm_scale
    scores = torch.relu(scores)
    scores = (scores * w.float().unsqueeze(-1)).sum(dim=2)
    mask = ratio_causal_mask(q.shape[1], k.shape[1], ratio, q.device).unsqueeze(0)
    return scores.masked_fill(~mask, float("-inf"))


DTYPE = torch.bfloat16


def ratio_causal_mask(seq_q, seq_k, ratio, device):
    """Valid iff k < (q + 1) // ratio."""
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
    return k_idx < ((q_idx + 1) // ratio)


def make_indexer_inputs(batch=2, seq_q=8, seq_k=16, heads=4, head_dim=64):
    generator = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn(
        batch,
        seq_q,
        heads,
        head_dim,
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    )
    k = torch.randn(
        batch,
        seq_k,
        1,
        head_dim,
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    )
    w = torch.randn(
        batch, seq_q, heads, device="cuda", generator=generator, dtype=torch.float32
    )
    return q.to(DTYPE), k.to(DTYPE), w.to(DTYPE)


HEAD_DIM = 512


def make_sparse_attn_inputs(
    tokens=64, heads=8, head_dim=HEAD_DIM, kv_tokens=128, topk=16
):
    generator = torch.Generator(device="cuda").manual_seed(5)
    q = torch.randn(
        tokens, heads, head_dim, device="cuda", generator=generator, dtype=torch.float32
    )
    kv = torch.randn(
        kv_tokens, head_dim, device="cuda", generator=generator, dtype=torch.float32
    )
    attn_sink = (
        torch.randn(heads, device="cuda", generator=generator, dtype=torch.float32)
        * 0.1
    )
    topk_idxs = torch.randint(
        0,
        kv_tokens,
        (tokens, topk),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    )
    return q.to(DTYPE), kv.to(DTYPE), attn_sink, topk_idxs


def test_indexer_fwd_matches_reference():
    q, k, w = make_indexer_inputs()
    scores = flash_train.torch.indexer_fwd(q, k, w, ratio=4, sm_scale=1.0)
    expected = ref_indexer_fwd(q, k, w, 4, 1.0)
    torch.testing.assert_close(scores, expected, rtol=1e-2, atol=1e-2, equal_nan=True)


def test_rejects_cpu_tensor():
    q, k, w = make_indexer_inputs()
    with pytest.raises(RuntimeError):
        flash_train.torch.indexer_fwd(q.cpu(), k.cpu(), w.cpu())


def test_rejects_unsupported_dtype_chain():
    q, k, w = make_indexer_inputs()
    with pytest.raises(RuntimeError):
        flash_train.torch.indexer_fwd(q, k.to(torch.int32), w)


def test_rejects_inconsistent_shapes():
    q, k, w = make_indexer_inputs()
    with pytest.raises(RuntimeError):
        flash_train.torch.indexer_fwd(q, k[:, : k.shape[1] // 2], w)


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
        return flash_train.torch.indexer_fwd(q, k, w, ratio=1, sm_scale=0.2, **metadata)

    expected = run(q.float(), k.float(), w.float())
    # Repeat the same signature so both cold selection and cache reuse execute.
    for _ in range(2):
        actual = run(q, k, w)
        torch.testing.assert_close(
            actual, expected, rtol=2e-4, atol=2e-5, equal_nan=True
        )
    with pytest.raises(RuntimeError):
        run(q, k.to(torch.int32), w)


def test_mixed_storage_uses_current_stream_and_replays_graph():
    q, k, w = make_indexer_inputs(batch=1, seq_q=8, seq_k=8, head_dim=32)
    w = w.float()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = flash_train.torch.indexer_fwd(q.float(), k.float(), w, ratio=1)
        warmup = flash_train.torch.indexer_fwd(q, k, w, ratio=1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(warmup, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flash_train.torch.indexer_fwd(q, k, w, ratio=1)
    q.mul_(0.5)
    expected = flash_train.torch.indexer_fwd(q.float(), k.float(), w, ratio=1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected)


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
    p["q"] = (
        torch.rand((1, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["k"] = (
        torch.rand((1, a, 1, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["w"] = (torch.rand((1, a, b), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["ratio"] = 1
    p["sm_scale"] = 0.25
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])[0]
    actual = flash_train.torch.indexer_fwd(
        q=p["q"], k=p["k"], w=p["w"], ratio=p["ratio"], sm_scale=p["sm_scale"]
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=rtol, atol=atol, equal_nan=True
    )
