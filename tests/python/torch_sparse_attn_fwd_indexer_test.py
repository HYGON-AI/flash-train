# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

HEAD_DIM = 512

DTYPE = torch.bfloat16


def ref_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, softmax_scale, topk_length=None):
    """Sink-inclusive softmax over the selected keys; KV-only LSE returned."""
    gathered_k = kv.float()[topk_idxs.clamp(min=0).long()]
    gathered_v = gathered_k[..., :HEAD_DIM]
    scores = torch.einsum("thd,tkd->thk", q.float(), gathered_k) * softmax_scale
    valid = topk_idxs >= 0
    if topk_length is not None:
        positions = torch.arange(topk_idxs.shape[-1], device=q.device).unsqueeze(0)
        valid = valid & (positions < topk_length.unsqueeze(1))
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    sink = attn_sink.view(1, -1, 1).expand(q.shape[0], q.shape[1], 1)
    logits = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., : topk_idxs.shape[-1]]
    out = torch.einsum("thk,tkd->thd", probs, gathered_v)
    return out.to(q.dtype), lse


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


def test_sparse_attn_fwd_indexer_matches_reference():
    q, kv, attn_sink, topk_idxs = make_sparse_attn_inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    indexer_topk = 8
    out, max_logits, lse, lse_indexer = flash_train.torch.sparse_attn_fwd_indexer(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale, indexer_topk=indexer_topk
    )
    expected_out, expected_lse = ref_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, scale)
    torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)

    # max_logits covers every selected entry; lse_indexer covers the first
    # indexer_topk entries, merged with the attention sink.
    gathered_all = kv.float()[topk_idxs.clamp(min=0).long()]
    scores_all = torch.einsum("thd,tkd->thk", q.float(), gathered_all) * scale
    # The selection runs along the last axis, not the head axis.
    indexer_lse = torch.logsumexp(scores_all[..., :indexer_topk], dim=-1)
    expected_lse_indexer = torch.logaddexp(indexer_lse, attn_sink.view(1, -1))
    torch.testing.assert_close(lse_indexer, expected_lse_indexer, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        max_logits, scores_all.amax(dim=-1), rtol=2e-2, atol=2e-2
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
    q, kv, sink, indices = make_sparse_attn_inputs(
        tokens=4, heads=4, kv_tokens=8, topk=8
    )
    q, kv = q.to(types[0]), kv.to(types[1])
    scale = 0.04

    def run(q, kv):
        return flash_train.torch.sparse_attn_fwd_indexer(
            q, kv, sink, indices, softmax_scale=scale, indexer_topk=4
        )

    expected = run(q.float(), kv.float())
    for _ in range(2):
        actual = run(q, kv)
        for i, (value, reference) in enumerate(zip(actual, expected)):
            dtype = q.dtype if i == 0 else torch.float32
            assert value.dtype == dtype
            torch.testing.assert_close(value, reference.to(dtype), rtol=2e-2, atol=2e-4)
    with pytest.raises(RuntimeError):
        run(q, kv.to(torch.int32))


# Non-power-of-two boundaries and every supported storage precision.
SHAPE_GRID = [(8, 8, 512), (33, 8, 512), (128, 16, 512)]
GRID_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Tolerances cover reduction error and FP16/BF16 storage rounding for
# inputs bounded to [-0.1, 0.1]; FP32 outputs retain the stricter tolerance.
GRID_TOLERANCES = {
    torch.float32: (0.0002, 2e-05),
    torch.float16: (0.02, 0.002),
    torch.bfloat16: (0.05, 0.005),
}


def torch_reference_sparse_attention(q, kv, sink, indices, scale):
    keys = kv.float()[indices.long()]
    logits = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    lse = torch.logsumexp(logits, -1)
    weights = torch.cat(
        (logits, sink.float()[None, :, None].expand(q.shape[0], -1, -1)), -1
    ).softmax(-1)[..., :-1]
    output = torch.einsum("thk,tkd->thd", weights, keys)
    return output, lse, logits


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["kv"] = (torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["attn_sink"] = (
        torch.rand((b,), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["topk_indices"] = (
        (torch.arange(128, device="cuda", dtype=torch.int32) % a)
        .expand((a, 128))
        .contiguous()
    )
    p["topk_length"] = torch.full((a,), 128, device="cuda", dtype=torch.int32)
    p["softmax_scale"] = 0.125
    p["indexer_topk"] = 32
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    values = torch_reference_sparse_attention(
        p["q"], p["kv"], p["attn_sink"], p["topk_indices"], p["softmax_scale"]
    )
    expected = (
        values[0].to(p["q"].dtype),
        values[2].amax(-1),
        values[1],
        torch.logaddexp(
            torch.logsumexp(values[2][..., : p["indexer_topk"]], -1),
            p["attn_sink"][None],
        ),
    )
    actual = flash_train.torch.sparse_attn_fwd_indexer(
        q=p["q"],
        kv=p["kv"],
        attn_sink=p["attn_sink"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
        indexer_topk=p["indexer_topk"],
        topk_length=p["topk_length"],
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
