# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


def ref_sparse_indexer_predict(q, k, w, topk_idxs, sm_scale):
    """softmax over the selected entries of the indexer score."""
    # Per-batch indexing: this DTK torch build mis-shapes multi-dimensional
    # fancy indexing against a three-dimensional input, so every gather below
    # indexes a two-dimensional matrix with a two-dimensional index.
    gathered = torch.stack(
        [k[b].float()[topk_idxs[b].clamp(min=0).long()] for b in range(q.shape[0])]
    )
    # Broadcasted multiply-and-sum rather than einsum: the HCU torch build's
    # einsum kernel faults on this operand set, and the explicit form is the
    # same arithmetic.
    scores = (q.float().unsqueeze(3) * gathered.unsqueeze(2)).sum(-1) * sm_scale
    scores = torch.relu(scores)
    scores = (scores * w.float().unsqueeze(-1)).sum(dim=2)
    return torch.softmax(scores, dim=-1)


DTYPE = torch.bfloat16


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


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def test_sparse_indexer_predict_matches_reference():
    q, k, w = make_indexer_inputs()
    generator = torch.Generator(device="cuda").manual_seed(3)
    topk_idxs = torch.randint(
        0, k.shape[1], (2, 8, 6), device="cuda", generator=generator, dtype=torch.int32
    )
    predict = flash_train.torch.sparse_indexer_predict(
        q, k.squeeze(2), w, topk_idxs, sm_scale=1.0
    )
    expected = ref_sparse_indexer_predict(q, k.squeeze(2), w, topk_idxs, 1.0)
    torch.testing.assert_close(predict, expected, rtol=2e-2, atol=2e-2)


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
        return flash_train.torch.sparse_indexer_predict(q, k, w, indices, sm_scale=0.2)

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


def torch_reference_gathered(q, k, indices, scale):
    # Index a 2-D matrix per batch: this HCU Torch build mis-shapes a
    # multidimensional fancy index against a batched 3-D matrix.
    keys = torch.stack([k[b].float()[indices[b].long()] for b in range(q.shape[0])])
    logits = (q.float().unsqueeze(3) * keys.unsqueeze(2)).sum(-1) * scale
    return keys, logits


def torch_reference_sparse_predict(q, k, w, indices, scale):
    _, logits = torch_reference_gathered(q, k, indices, scale)
    return (logits.relu() * w.float().unsqueeze(-1)).sum(2).softmax(-1)


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
    p["w"] = (torch.rand((1, a, b), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["topk_indices"] = (
        (torch.arange(min(a, 32), device="cuda", dtype=torch.int32) % a)
        .expand((1, a, min(a, 32)))
        .contiguous()
    )
    p["sm_scale"] = 0.25
    p["topk_indices_global"] = 0
    return p


@pytest.mark.parametrize("shape", SHAPE_GRID)
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_matches_torch(shape, dtype):
    p = make_numerics_inputs(shape, dtype)
    expected = torch_reference_sparse_predict(
        p["q"], p["k"], p["w"], p["topk_indices"], p["sm_scale"]
    )
    actual = flash_train.torch.sparse_indexer_predict(
        q=p["q"],
        k=p["k"],
        w=p["w"],
        topk_idxs=p["topk_indices"],
        sm_scale=p["sm_scale"],
        topk_indices_global=p["topk_indices_global"],
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=rtol, atol=atol, equal_nan=True
    )
