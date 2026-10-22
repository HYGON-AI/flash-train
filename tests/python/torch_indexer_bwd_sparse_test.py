# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flash_train.torch

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


@pytest.mark.parametrize("k_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_indexer_bwd_sparse_matches_upstream(k_dtype):
    upstream = pytest.importorskip(
        "hip_dsa_kernels", reason="upstream hip_dsa_kernels is not importable"
    )
    q, k, w = make_indexer(seq_k=16)
    k = k.to(k_dtype)
    index_k = k.contiguous()
    generator = torch.Generator(device="cuda").manual_seed(21)
    topk = torch.randint(
        0,
        index_k.shape[1],
        (*q.shape[:2], 6),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    )
    attn_score = torch.rand(*topk.shape, device="cuda")
    index_score = torch.softmax(torch.randn(*topk.shape, device="cuda"), dim=-1)
    theirs = upstream.indexer_bwd_sparse(
        q, w, index_k, attn_score.clone(), index_score, topk, 0.5, 1.0, 1.0, False
    )
    ours = flash_train.torch.indexer_bwd_sparse(
        q,
        w,
        index_k,
        attn_score.clone(),
        index_score,
        topk,
        sm_scale=0.5,
        loss_coeff=1.0,
        grad_loss=1.0,
    )
    torch.testing.assert_close(ours[1], theirs[0], equal_nan=False)
    torch.testing.assert_close(ours[2], theirs[1], equal_nan=False)
    torch.testing.assert_close(ours[3], theirs[2], equal_nan=False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("head_dim", [32, 128, 576])
@pytest.mark.parametrize("cross_batch", [False, True])
def test_global_sparse_gather_and_gradient_scatter(dtype, head_dim, cross_batch):
    import flash_train.torch as ft

    torch.manual_seed(528)
    batch, tokens, heads, keys, topk = 2, 4, 32, 12, 8
    q = torch.rand(batch, tokens, heads, head_dim, device="cuda", dtype=dtype) * 0.1
    kv = torch.rand(batch, keys, head_dim, device="cuda", dtype=dtype) * 0.1
    weights = torch.rand(batch, tokens, heads, device="cuda", dtype=dtype)
    indices = torch.randint(
        keys, (batch, tokens, topk), device="cuda", dtype=torch.int32
    )
    owner = torch.tensor(
        [1, 0] if cross_batch else [0, 1], device="cuda", dtype=torch.int32
    )
    global_indices = indices + owner[:, None, None] * keys
    reference_kv = kv.flip(0).contiguous() if cross_batch else kv
    lse = torch.randn(batch, tokens, heads, device="cuda")
    operations = [(ft.sparse_attn_target, lse)]
    # Indexer kernels support D <= 128; attention target also supports MLA D=576.
    if head_dim <= 128:
        operations.append((ft.sparse_indexer_predict, weights))
    for operation, extra in operations:
        expected = operation(q, reference_kv, extra, indices)
        actual = operation(q, kv, extra, global_indices, topk_indices_global=True)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    if head_dim > 128:
        return
    target = torch.softmax(torch.randn(batch, tokens, topk, device="cuda"), dim=-1)
    predict = torch.softmax(torch.randn_like(target), dim=-1)
    expected = ft.indexer_bwd_sparse(
        q, weights, reference_kv, target.clone(), predict, indices
    )
    actual = ft.indexer_bwd_sparse(
        q,
        weights,
        kv,
        target.clone(),
        predict,
        global_indices,
        topk_indices_global=True,
    )
    # The last result is dK: cross-batch gathers must scatter back to the key owner.
    expected = list(expected)
    if cross_batch:
        expected[-1] = expected[-1].flip(0)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-4)


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
    target = torch.rand(1, 8, 4, device="cuda")
    predict = torch.softmax(torch.randn_like(target), -1)

    def run(q, k, w):
        return flash_train.torch.indexer_bwd_sparse(
            q, w, k, target.clone(), predict, indices, sm_scale=0.2
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


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize(
    "topk, selection_start",
    [(512, 0), (512, 192), (512, 480), (2048, 0), (2048, 192), (2048, 2016)],
)
@pytest.mark.parametrize("device_grad_loss", [False, True])
def test_large_sparse_backward_fast_path(
    heads, topk, selection_start, device_grad_loss
):
    q, k, w = make_indexer(batch=1, seq_q=4096, seq_k=4096, heads=heads, head_dim=128)
    q, k, w = q * 0.1, k * 0.1, w * 0.1
    indices = torch.full((1, 4096, topk), -1, device="cuda", dtype=torch.int32)
    # Late selections exercise all four bound slots before score-gradient scratch reuse.
    selected = slice(selection_start, selection_start + 32)
    indices[:, :, selected] = torch.arange(32, device="cuda", dtype=torch.int32)
    target = torch.zeros(indices.shape, device="cuda")
    predict = torch.zeros_like(target)
    target[:, :, selected] = torch.softmax(torch.randn(1, 4096, 32, device="cuda"), -1)
    predict[:, :, selected] = torch.softmax(torch.randn(1, 4096, 32, device="cuda"), -1)
    grad_loss = torch.tensor(4096.0, device="cuda") if device_grad_loss else 4096.0
    result = flash_train.torch.indexer_bwd_sparse(
        q, w, k, target.clone(), predict, indices, sm_scale=0.5, grad_loss=grad_loss
    )
    # Same mathematical selection with no padded entries selects the general path.
    expected = flash_train.torch.indexer_bwd_sparse(
        q,
        w,
        k,
        target[:, :, selected].contiguous(),
        predict[:, :, selected].contiguous(),
        indices[:, :, selected].contiguous(),
        sm_scale=0.5,
        grad_loss=grad_loss,
    )
    for actual, reference in zip(result[1:], expected[1:]):
        torch.testing.assert_close(actual, reference, rtol=5e-2, atol=2e-3)


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


def torch_reference_sparse_predict(q, k, w, indices, scale):
    _, logits = torch_reference_gathered(q, k, indices, scale)
    return (logits.relu() * w.float().unsqueeze(-1)).sum(2).softmax(-1)


def torch_reference_indexer_sparse_backward(p):
    q, w, k = (p[n].float() for n in ("index_q", "weights", "index_k"))
    keys, raw = torch_reference_gathered(q, k, p["topk_indices"], p["sm_scale"])
    target, probability = p["attn_score"].float(), p["index_score"].float()
    direct = (
        -target.clamp_min(math.exp(-100)) * (probability >= math.exp(-100)) / q.shape[1]
    )
    direct = direct * p["loss_coeff"] * p["grad_loss"]
    ds = direct - probability * direct.sum(-1, keepdim=True)
    dw = (raw.relu() * ds.unsqueeze(2)).sum(-1)
    dg = ds.unsqueeze(2) * w.unsqueeze(-1) * (raw > 0) * p["sm_scale"]
    dq = (dg.unsqueeze(-1) * keys.unsqueeze(2)).sum(3)
    contribution = (dg.unsqueeze(-1) * q.unsqueeze(3)).sum(2)
    dk = torch.zeros_like(k)
    for b in range(q.shape[0]):
        dk[b].index_add_(
            0, p["topk_indices"][b].flatten().long(), contribution[b].flatten(0, 1)
        )
    return (
        ds,
        dq.to(p["index_q"].dtype),
        dw.to(p["weights"].dtype),
        dk.to(p["index_k"].dtype),
    )


def make_numerics_inputs(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["index_q"] = (
        torch.rand((1, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["weights"] = (
        torch.rand((1, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["index_k"] = (
        torch.rand((1, a, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["attn_score"] = (
        torch.rand((1, a, min(a, 32)), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["index_score"] = (
        torch.rand((1, a, min(a, 32)), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["topk_indices"] = (
        (torch.arange(min(a, 32), device="cuda", dtype=torch.int32) % a)
        .expand((1, a, min(a, 32)))
        .contiguous()
    )
    p["sm_scale"] = 0.25
    p["loss_coeff"] = 1
    p["grad_loss"] = 1
    p["topk_indices_global"] = 0
    p["attn_score"] = p["attn_score"].abs()
    p["index_score"] = torch_reference_sparse_predict(
        p["index_q"], p["index_k"], p["weights"], p["topk_indices"], p["sm_scale"]
    )
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    # Non-unit loss scaling keeps every gradient above the absolute error budget.
    p["loss_coeff"] = 0.5
    p["grad_loss"] = 512.0 * shape[0]
    expected = torch_reference_indexer_sparse_backward(p)
    actual = flash_train.torch.indexer_bwd_sparse(
        index_q=p["index_q"],
        weights=p["weights"],
        index_k=p["index_k"],
        attn_score=p["attn_score"],
        index_score=p["index_score"],
        topk_idxs=p["topk_indices"],
        sm_scale=p["sm_scale"],
        loss_coeff=p["loss_coeff"],
        grad_loss=p["grad_loss"],
        topk_indices_global=p["topk_indices_global"],
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


@pytest.mark.parametrize("heads,head_dim,topk", [(4, 64, 6), (64, 128, 128)])
def test_device_grad_loss_changes_on_graph_replay(heads, head_dim, topk):
    q, k, w = make_indexer(batch=1, seq_q=16, seq_k=128, heads=heads, head_dim=head_dim)
    indices = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .expand(1, 16, topk)
        .contiguous()
    )
    target = torch.rand(1, 16, topk, device="cuda")
    predict = torch.softmax(torch.randn_like(target), dim=-1)
    score = target.clone()
    grad_loss = torch.ones((), device="cuda", dtype=torch.float32)

    def backward(scale):
        return flash_train.torch.indexer_bwd_sparse(
            q,
            w,
            k,
            score,
            predict,
            indices,
            sm_scale=0.5,
            loss_coeff=0.1,
            grad_loss=scale,
        )

    backward(grad_loss)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        score.copy_(target)
        actual = backward(grad_loss)
    for value in (0.25, 1.5):
        score.copy_(target)
        expected = tuple(t.clone() for t in backward(value))
        grad_loss.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0.02, atol=0.01)


def test_device_grad_loss_does_not_read_back_to_host():
    q, k, w = make_indexer()
    indices = torch.zeros(2, 8, 4, device="cuda", dtype=torch.int32)
    score = torch.rand(2, 8, 4, device="cuda")
    predict = torch.softmax(torch.randn_like(score), dim=-1)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        flash_train.torch.indexer_bwd_sparse(
            q, w, k, score, predict, indices, grad_loss=scale
        )
    assert not any(event.key == "aten::item" for event in profile.key_averages())
