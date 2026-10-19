# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

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
    metadata = dict(cu_seqlens_q=cu, cu_seqlens_k=cu)
    k = k.squeeze(-2)
    target = torch.rand(8, 4, device="cuda")
    norm = torch.ones_like(target[..., 0])
    predict = torch.softmax(torch.randn_like(target), -1)

    def run(q, k, w):
        return flash_train.torch.indexer_bwd_dense_thd(
            q,
            w,
            k,
            target,
            norm,
            predict.clone(),
            norm,
            ratio=1,
            sm_scale=0.2,
            **metadata,
        )

    expected = run(q.float(), k.float(), w.float())
    # Repeat the same signature so both cold selection and cache reuse execute.
    for _ in range(2):
        actual = run(q, k, w)
        for value, reference, dtype in zip(
            actual, expected, [torch.float32, q.dtype, w.dtype, k.dtype]
        ):
            assert value.dtype == dtype
            torch.testing.assert_close(
                value, reference.to(dtype), rtol=2e-2, atol=2e-4, equal_nan=False
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


def torch_reference_indexer_dense_backward(p):
    packed = p["index_q"].ndim == 3
    q, w, k = (p[n].float() for n in ("index_q", "weights", "index_k"))
    teacher, norm, logits, lse = (
        p[n].float() for n in ("attn_score", "attn_l1norm", "index_score", "index_lse")
    )
    if packed:
        q, w, k, teacher, norm, logits, lse = (
            t[None] for t in (q, w, k, teacher, norm, logits, lse)
        )
    valid = torch_reference_mask(q.shape[1], k.shape[1], q.device)[None]
    target = (teacher / (norm.unsqueeze(-1) + 1e-10)).clamp_min(math.exp(-100))
    log_p = logits - lse.unsqueeze(-1)
    direct = -(target * (log_p >= -100)).masked_fill(~valid, 0) / q.shape[1]
    ds = (direct - log_p.exp() * direct.sum(-1, keepdim=True)).masked_fill(~valid, 0)
    raw = torch.einsum("bqhd,bkd->bqhk", q, k) * p["sm_scale"]
    dw = (raw.relu() * ds.unsqueeze(2)).sum(-1)
    dg = ds.unsqueeze(2) * w.unsqueeze(-1) * (raw > 0) * p["sm_scale"]
    dq = torch.einsum("bqhk,bkd->bqhd", dg, k)
    dk = torch.einsum("bqhk,bqhd->bkd", dg, q)
    outputs = (ds, dq, dw, dk)
    if packed:
        outputs = tuple(t[0] for t in outputs)
    return (
        outputs[0],
        *(
            t.to(p[n].dtype)
            for t, n in zip(outputs[1:], ("index_q", "weights", "index_k"))
        ),
    )


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["index_q"] = (
        torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["weights"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["index_k"] = (
        torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["attn_score"] = (
        torch.rand((a, a), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["attn_l1norm"] = (
        torch.rand((a,), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["index_score"] = (
        torch.rand((a, a), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["index_lse"] = (
        torch.rand((a,), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["cu_seqlens_q"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["cu_seqlens_k"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["q_causal_offsets"] = torch.zeros((1,), device="cuda", dtype=torch.int32)
    p["sm_scale"] = 0.25
    p["loss_coeff"] = 1
    p["grad_loss"] = 1
    p["ratio"] = 1
    p["index_score"], p["index_lse"] = torch_reference_dense_indexer(
        p["index_q"], p["index_k"].unsqueeze(-2), p["weights"], p["sm_scale"]
    )
    valid = torch_reference_mask(a, a, p["index_q"].device)
    p["attn_score"] = p["attn_score"].abs().masked_fill(~valid, 0)
    p["attn_l1norm"] = p["attn_score"].sum(-1)
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_indexer_dense_backward(p)
    actual = flash_train.torch.indexer_bwd_dense_thd(
        index_q=p["index_q"],
        weights=p["weights"],
        index_k=p["index_k"],
        attn_score=p["attn_score"],
        attn_l1norm=p["attn_l1norm"],
        index_score=p["index_score"],
        index_lse=p["index_lse"],
        cu_seqlens_q=p["cu_seqlens_q"],
        cu_seqlens_k=p["cu_seqlens_k"],
        sm_scale=p["sm_scale"],
        loss_coeff=p["loss_coeff"],
        grad_loss=p["grad_loss"],
        ratio=p["ratio"],
        q_causal_offsets=p["q_causal_offsets"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol, equal_nan=False
        )
