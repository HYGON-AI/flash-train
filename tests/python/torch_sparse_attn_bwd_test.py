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


def test_sparse_attn_bwd_matches_autograd():
    q, kv, attn_sink, topk_idxs = make_sparse_attn_inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale
    )
    generator = torch.Generator(device="cuda").manual_seed(9)
    dout = torch.randn(
        out.shape, device="cuda", generator=generator, dtype=torch.float32
    ).to(DTYPE)
    dq, dkv, dsink = flash_train.torch.sparse_attn_bwd(
        q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale=scale
    )

    reference_q = q.float().detach().requires_grad_(True)
    reference_kv = kv.float().detach().requires_grad_(True)
    reference_sink = attn_sink.detach().requires_grad_(True)
    reference_out, _ = ref_sparse_attn_fwd(
        reference_q, reference_kv, reference_sink, topk_idxs, scale
    )
    (reference_out * dout.float()).sum().backward()

    torch.testing.assert_close(dq.float(), reference_q.grad, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(dkv.float(), reference_kv.grad, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(dsink, reference_sink.grad, rtol=5e-2, atol=2e-2)


def test_wrapper_namespace_matches_the_pure_operators():
    from flash_train.torch import DSA, sparse_attn_forward

    q, kv, attn_sink, topk_idxs = make_sparse_attn_inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale
    )
    adapter_out, adapter_lse = sparse_attn_forward(
        q, kv, attn_sink, topk_idxs, softmax_scale=scale
    )
    torch.testing.assert_close(adapter_out, out)
    torch.testing.assert_close(adapter_lse, lse)
    result = DSA.sparse_attention_backward_wrapper(
        q, kv, out, out, lse, attn_sink, topk_idxs, softmax_scale=scale
    )
    assert set(result) == {"dq", "dkv", "d_sink"}
    assert result["dq"].shape == q.shape


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
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=scale
    )
    dout = torch.randn_like(out)

    def run(q, kv, out, dout):
        return flash_train.torch.sparse_attn_bwd(
            q, kv, out, dout, lse, sink, indices, softmax_scale=scale
        )

    expected = run(q.float(), kv.float(), out.float(), dout.float())
    for _ in range(2):
        actual = run(q, kv, out, dout)
        for value, reference, dtype in zip(
            actual, expected, [q.dtype, kv.dtype, torch.float32]
        ):
            assert value.dtype == dtype
            torch.testing.assert_close(value, reference.to(dtype), rtol=2e-2, atol=2e-4)
    with pytest.raises(RuntimeError):
        run(q, kv, out.float() if q.dtype != torch.float32 else out.half(), dout)


@pytest.mark.parametrize(
    "tokens,heads,padded",
    [
        (1024, 8, False),
        (1024, 16, False),
        (4096, 32, False),
        (4096, 64, False),
        (4097, 64, True),
    ],
)
@pytest.mark.parametrize("workspace_mode", ["0", "1"])
def test_d576_backward_inverse_workspace(
    tokens, heads, padded, workspace_mode, monkeypatch
):
    monkeypatch.setenv("DSA_BWD_WORKSPACE", workspace_mode)
    q, kv, sink, _ = make_sparse_attn_inputs(
        tokens=tokens, heads=heads, head_dim=576, kv_tokens=4096, topk=32
    )
    q = q * 0.1
    kv = kv * 0.1
    live_heads = 32 if padded else heads
    if padded:
        q[:, 32:] = 0
        sink[32:] = float("-inf")
    indices = torch.full((tokens, 2048), -1, device="cuda", dtype=torch.int32)
    indices[:, :96] = torch.arange(96, device="cuda", dtype=torch.int32)
    scale = 576**-0.5
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=scale
    )
    dout = torch.randn_like(out) * 0.1
    if padded:
        out[:, 32:] = 0
        dout[:, 32:] = 0
        lse[:, 32:] = 0
    dq, dkv, dsink = flash_train.torch.sparse_attn_bwd(
        q, kv, out, dout, lse, sink, indices, softmax_scale=scale
    )
    # A small valid selection keeps an independent full-gradient reference affordable.
    qr = q[:, :live_heads].float().detach().requires_grad_(True)
    kr = kv[:96].float().detach().requires_grad_(True)
    sr = sink[:live_heads].detach().requires_grad_(True)
    scores = torch.einsum("thd,kd->thk", qr, kr) * scale
    probs = torch.softmax(
        torch.cat([scores, sr[None, :, None].expand(tokens, -1, 1)], -1), -1
    )[..., :96]
    reference = torch.einsum("thk,kd->thd", probs, kr[:, :512])
    (reference * dout[:, :live_heads].float()).sum().backward()
    torch.testing.assert_close(
        dq[:, :live_heads].float(), qr.grad, rtol=6e-2, atol=2e-3
    )
    torch.testing.assert_close(dkv[:96].float(), kr.grad, rtol=6e-2, atol=3e-2)
    torch.testing.assert_close(dsink[:live_heads], sr.grad, rtol=6e-2, atol=3e-3)
    assert torch.count_nonzero(dkv[96:]) == 0
    if padded:
        assert torch.count_nonzero(dq[:, 32:]) == 0
        assert torch.count_nonzero(dsink[32:]) == 0


@pytest.mark.parametrize("mode", [None, "", "0", "1", "2", "invalid", " 1 "])
def test_backward_workspace_environment_validation(mode, monkeypatch):
    if mode is None:
        monkeypatch.delenv("DSA_BWD_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("DSA_BWD_WORKSPACE", mode)
    q, kv, sink, indices = make_sparse_attn_inputs(
        tokens=4, heads=4, kv_tokens=8, topk=4
    )
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=0.04
    )

    def run():
        return flash_train.torch.sparse_attn_bwd(
            q, kv, out, torch.zeros_like(out), lse, sink, indices, softmax_scale=0.04
        )

    if mode not in (None, "", "0", "1"):
        with pytest.raises(RuntimeError, match="failed with status 1"):
            run()
    else:
        for value in run():
            assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cached_backward_switch_does_not_disable_other_precisions(dtype, monkeypatch):
    monkeypatch.setenv("DSA_SAB2_CACHED", "0")
    q, kv, sink, indices = make_sparse_attn_inputs(
        tokens=4, heads=16, head_dim=512, kv_tokens=128, topk=128
    )
    q, kv = q.to(dtype), kv.to(dtype)
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=0.04
    )
    for value in flash_train.torch.sparse_attn_bwd(
        q, kv, out, torch.zeros_like(out), lse, sink, indices, softmax_scale=0.04
    ):
        assert torch.count_nonzero(value) == 0


# Non-power-of-two boundaries and every supported storage precision.
SHAPE_GRID = [(8, 8, 512), (33, 8, 512), (128, 16, 512)]
GRID_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Tolerances cover reduction error and FP16/BF16 storage rounding for
# inputs bounded to [-0.1, 0.1], including the stored forward output.
# Absolute budgets stay below every grid reference gradient magnitude.
GRID_TOLERANCES = {
    torch.float32: (0.0002, 2e-05),
    torch.float16: (0.02, 5e-5),
    torch.bfloat16: (0.05, 2e-4),
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


def torch_reference_sparse_attention_backward(p):
    q, kv, sink = (
        p[n].detach().float().requires_grad_() for n in ("q", "kv", "attn_sink")
    )
    output, _, _ = torch_reference_sparse_attention(
        q, kv, sink, p["topk_indices"], p["softmax_scale"]
    )
    grads = torch.autograd.grad(output, (q, kv, sink), p["dout"].float())
    return tuple(t.to(p[n].dtype) for t, n in zip(grads, ("q", "kv", "attn_sink")))


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
    p["out"] = (
        torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["dout"] = (
        torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["lse"] = (torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.float32
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
    out, lse, _ = torch_reference_sparse_attention(
        p["q"], p["kv"], p["attn_sink"], p["topk_indices"], p["softmax_scale"]
    )
    p["out"] = out.to(p["q"].dtype)
    p["lse"] = lse
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_sparse_attention_backward(p)
    actual = flash_train.torch.sparse_attn_bwd(
        q=p["q"],
        kv=p["kv"],
        out=p["out"],
        dout=p["dout"],
        lse=p["lse"],
        attn_sink=p["attn_sink"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("value_dim", [512, 576])
@pytest.mark.parametrize("cached", ["0", "1"])
@pytest.mark.parametrize("heads,topk", [(4, 8), (8, 128)])
def test_d576_backward_preserves_value_width(
    dtype, value_dim, cached, heads, topk, monkeypatch
):
    monkeypatch.setenv("DSA_SAB2_CACHED", cached)
    q, kv, sink, indices = make_sparse_attn_inputs(
        tokens=4, heads=heads, head_dim=576, kv_tokens=8, topk=topk
    )
    q, kv = q.to(dtype) * 0.1, kv.to(dtype) * 0.1
    rq = q.float().detach().requires_grad_()
    rk = kv.float().detach().requires_grad_()
    rs = sink.detach().requires_grad_()
    keys = rk[indices.long()]
    logits = torch.einsum("thd,tkd->thk", rq, keys) * 0.04
    probs = torch.cat((logits, rs[None, :, None].expand(4, -1, -1)), -1).softmax(-1)[
        ..., :-1
    ]
    output = torch.einsum("thk,tkd->thd", probs, keys[..., :value_dim])
    out = output.detach().to(dtype)
    dout = torch.randn_like(out) * 0.1
    expected = torch.autograd.grad(output, (rq, rk, rs), dout.float())
    backward = flash_train.torch.dsa_wrappers._sparse_attn_bwd_op

    def run_backward(**kwargs):
        # Guard the storage contract: zero-padding is numerically equivalent
        # but selects a slower kernel and allocates extra device buffers.
        assert kwargs["out"].shape[-1] == value_dim
        assert kwargs["dout"].shape[-1] == value_dim
        return backward(**kwargs)

    monkeypatch.setattr(
        flash_train.torch.dsa_wrappers, "_sparse_attn_bwd_op", run_backward
    )
    actual = flash_train.torch.DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        logits.detach().logsumexp(-1),
        sink,
        indices,
        softmax_scale=0.04,
    )
    for name, ref, storage in zip(
        ("dq", "dkv", "d_sink"), expected, (dtype, dtype, torch.float32)
    ):
        torch.testing.assert_close(actual[name], ref.to(storage), rtol=5e-2, atol=2e-3)
    # The last 64 dK columns receive P*dO only for genuinely full-width V.
    assert out.shape[-1] == value_dim


@pytest.mark.parametrize("cached", ["0", "1"])
def test_mixed_d576_compact_backward_matches_fp32(cached, monkeypatch):
    monkeypatch.setenv("DSA_SAB2_CACHED", cached)
    q, kv, sink, indices = make_sparse_attn_inputs(
        tokens=4, heads=4, head_dim=576, kv_tokens=8, topk=8
    )
    q, kv = q * 0.1, kv.float() * 0.1
    out, lse = flash_train.torch.sparse_attn_fwd(
        q, kv, sink, indices, softmax_scale=0.04
    )
    dout = torch.randn_like(out) * 0.1
    expected = flash_train.torch.sparse_attn_bwd(
        q.float(), kv, out.float(), dout.float(), lse, sink, indices, softmax_scale=0.04
    )
    actual = flash_train.torch.DSA.sparse_attention_backward_wrapper(
        q, kv, out, dout, lse, sink, indices, softmax_scale=0.04
    )
    for name, ref, dtype in zip(
        ("dq", "dkv", "d_sink"), expected, (q.dtype, kv.dtype, sink.dtype)
    ):
        torch.testing.assert_close(actual[name], ref.to(dtype), rtol=5e-2, atol=2e-3)
