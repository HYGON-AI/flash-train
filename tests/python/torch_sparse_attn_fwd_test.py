# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from flash_train.torch import DSA
import flash_train.torch
import flash_train.torch as ft

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


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def test_sparse_attn_fwd_matches_reference():
    q, kv, attn_sink, topk_idxs = make_sparse_attn_inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale
    )
    expected_out, expected_lse = ref_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, scale)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)


def test_sparse_attn_fwd_with_topk_length_matches_reference():
    q, kv, attn_sink, topk_idxs = make_sparse_attn_inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    topk_length = torch.full(
        (q.shape[0],), topk_idxs.shape[-1] // 2, device="cuda", dtype=torch.int32
    )
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale, topk_length=topk_length
    )
    expected_out, expected_lse = ref_sparse_attn_fwd(
        q, kv, attn_sink, topk_idxs, scale, topk_length
    )
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out, expected_out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sparse_attention_wrapper_preserves_value_channels(dtype):
    q, kv = rand(4, 4, 576, dtype=dtype), rand(8, 576, dtype=dtype)
    sink = torch.zeros(4, device="cuda")
    indices = (
        torch.arange(8, device="cuda", dtype=torch.int32).expand(4, 8).contiguous()
    )
    out, lse = ft.dsa_wrappers.sparse_attn_forward(
        q, kv, sink, indices, softmax_scale=0.04
    )
    grad = torch.ones_like(out)
    result = DSA.sparse_attention_backward_wrapper(
        q, kv, out, grad, lse, sink, indices, softmax_scale=0.04
    )
    expected = ft.sparse_attn_bwd(
        q,
        kv,
        torch.nn.functional.pad(out, (0, 64)),
        torch.nn.functional.pad(grad, (0, 64)),
        lse,
        sink,
        indices,
        softmax_scale=0.04,
    )
    for name, value in zip(("dq", "dkv", "d_sink"), expected):
        torch.testing.assert_close(result[name], value, rtol=2e-2, atol=2e-2)


def test_sparse_attention_rejects_unaligned_head_dimension_during_selection():
    q, kv = rand(4, 4, 513), rand(8, 513)
    sink = torch.zeros(4, device="cuda")
    indices = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="status 4"):
        ft.sparse_attn_fwd(q, kv, sink, indices)


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
        return flash_train.torch.sparse_attn_fwd(
            q, kv, sink, indices, softmax_scale=scale
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


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
def test_d576_large_query_fast_path(heads):
    # T >= 1024 selects the MLA path; H <= 32 exercises the new two-wave tile.
    q, kv, sink, _ = make_sparse_attn_inputs(
        tokens=1024, heads=heads, head_dim=576, kv_tokens=256, topk=32
    )
    indices = torch.full((1024, 2048), -1, device="cuda", dtype=torch.int32)
    indices[:, :128] = torch.arange(128, device="cuda", dtype=torch.int32)
    scale = 576**-0.5
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=scale
    )
    rows = torch.tensor([0, 511, 1023], device="cuda")
    expected_out, expected_lse = ref_sparse_attn_fwd(
        q[rows], kv, sink, indices[rows, :128], scale
    )
    torch.testing.assert_close(out[rows], expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse[rows], expected_lse, rtol=2e-2, atol=2e-2)
    indexed = flash_train.torch.sparse_attn_fwd_indexer(
        q, kv, sink, indices, indexer_topk=128, softmax_scale=scale
    )
    torch.testing.assert_close(indexed[0], out)
    torch.testing.assert_close(indexed[2], lse)
    torch.testing.assert_close(
        indexed[3][rows], torch.logaddexp(expected_lse, sink), rtol=2e-2, atol=2e-2
    )


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
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    values = torch_reference_sparse_attention(
        p["q"], p["kv"], p["attn_sink"], p["topk_indices"], p["softmax_scale"]
    )
    expected = (values[0].to(p["q"].dtype), values[1])
    actual = flash_train.torch.sparse_attn_fwd(
        q=p["q"],
        kv=p["kv"],
        attn_sink=p["attn_sink"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
        topk_length=p["topk_length"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol, equal_nan=True
        )
