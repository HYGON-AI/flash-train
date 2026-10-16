# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)


def ratio_causal_mask(seq_q, seq_k, ratio, device):
    """Valid iff k < (q + 1) // ratio."""
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
    return k_idx < ((q_idx + 1) // ratio)


DTYPE = torch.bfloat16


def ref_indexer_fwd(q, k, w, ratio, sm_scale):
    """scores[b,q,k] = mask * sum_h ReLU(q.k^T * sm_scale) * w."""
    scores = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * sm_scale
    scores = torch.relu(scores)
    scores = (scores * w.float().unsqueeze(-1)).sum(dim=2)
    mask = ratio_causal_mask(q.shape[1], k.shape[1], ratio, q.device).unsqueeze(0)
    return scores.masked_fill(~mask, float("-inf"))


def ref_dense_indexer_score(q, k, w, ratio, sm_scale):
    """The same scores plus their log-sum-exp denominator."""
    scores = ref_indexer_fwd(q, k, w, ratio, sm_scale)
    return scores, torch.logsumexp(scores, dim=-1)


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


def test_dense_indexer_score_matches_reference():
    q, k, w = make_indexer_inputs()
    out, denom = flash_train.torch.dense_indexer_score(q, k, w, ratio=4, sm_scale=1.0)
    expected_out, expected_denom = ref_dense_indexer_score(q, k, w, 4, 1.0)
    torch.testing.assert_close(out, expected_out, rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(denom, expected_denom, rtol=1e-2, atol=1e-2)


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
        return flash_train.torch.dense_indexer_score(
            q, k, w, ratio=1, sm_scale=0.2, **metadata
        )

    expected = run(q.float(), k.float(), w.float())
    # Repeat the same signature so both cold selection and cache reuse execute.
    for _ in range(2):
        actual = run(q, k, w)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(
                value, reference, rtol=2e-4, atol=2e-5, equal_nan=True
            )
    with pytest.raises(RuntimeError):
        run(q, k.to(torch.int32), w)


@pytest.mark.parametrize(
    "dtype,heads,width,keys",
    [
        (torch.bfloat16, 2, 160, 4),  # No BF16 specialization.
        (torch.float32, 129, 32, 4),  # Head tile coverage.
        (torch.float16, 2, 32, 100000),  # Row scores exceed block LDS.
    ],
)
def test_launch_limits_raise_unsupported_before_execution(dtype, heads, width, keys):
    q = torch.zeros((1, 4, heads, width), device="cuda", dtype=dtype)
    k = torch.zeros((1, keys, 1, width), device="cuda", dtype=dtype)
    w = torch.ones((1, 4, heads), device="cuda", dtype=dtype)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="status 4"):
            flash_train.torch.dense_indexer_score(q, k, w, ratio=1, sm_scale=1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_runtime_width_survives_bf16_specialization_restriction(dtype):
    q = torch.full((1, 4, 2, 160), 0.25, device="cuda", dtype=dtype)
    k = torch.full((1, 4, 1, 160), 0.5, device="cuda", dtype=dtype)
    w = torch.ones((1, 4, 2), device="cuda", dtype=dtype)
    expected = ref_dense_indexer_score(q, k, w, ratio=1, sm_scale=0.25)
    for _ in range(2):
        actual = flash_train.torch.dense_indexer_score(q, k, w, ratio=1, sm_scale=0.25)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=2e-4, atol=2e-5)


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
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])
    actual = flash_train.torch.dense_indexer_score(
        q=p["q"], k=p["k"], w=p["w"], ratio=p["ratio"], sm_scale=p["sm_scale"]
    )
    rtol, atol = GRID_TOLERANCES[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.to(result.dtype), rtol=rtol, atol=atol, equal_nan=True
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
    p["k"] = (p["k"][:, :columns] * 10).to(k_dtype).contiguous()
    expected = torch_reference_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"])
    rtol, atol = GRID_TOLERANCES[q_dtype] if q_dtype == k_dtype else (2e-4, 2e-5)
    for _ in range(16):
        actual = flash_train.torch.dense_indexer_score(**p)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
