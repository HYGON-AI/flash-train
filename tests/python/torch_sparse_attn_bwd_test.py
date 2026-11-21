# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

# Deferred register-fused, inverse-workspace and cached-policy tests were
# removed with those implementations. Python tests cover default-path API and
# autograd behavior. Test every applicable Primitive in the C++ Plan-based
# numerical suite; keep configuration boundaries in white-box tests and use
# registry traversal for names/clone/common invalid precision. Do not add fixed
# candidate counts or a complete Primitive-name list here.

# Initial operator submissions retain one precision (BF16 by default).
# Other storage/mixed cases return with their implementations; keep Plan-based
# numerical enumeration and the configuration-boundary guidance above.
DTYPES = [torch.bfloat16]
# FP32 reductions and FP16/BF16 storage rounding, including the stored O.
TOLERANCES = {
    torch.bfloat16: (5e-2, 2e-4),
}


def reference_forward(q, kv, sink, indices, scale, value_dim, lengths=None):
    """Independent Torch forward: sink-inclusive O and KV-only LSE."""
    kv = kv.float().reshape(-1, kv.shape[-1])
    valid = (indices >= 0) & (indices < kv.shape[0])
    if lengths is not None:
        positions = torch.arange(indices.shape[-1], device=indices.device)
        valid = valid & (positions < lengths[:, None])
    keys = kv[indices.clamp(0, kv.shape[0] - 1).long()]
    logits = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    lse = logits.logsumexp(-1)
    scores = torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
    probabilities = scores.softmax(-1)[..., :-1]
    output = torch.einsum("thk,tkd->thd", probabilities, keys[..., :value_dim])
    return output, lse


def make_inputs(shape, q_type, kv_type=None, value_dim=None, lengths=False):
    tokens, heads, width, keys, topk = shape
    value_dim = width if value_dim is None else value_dim
    kv_type = q_type if kv_type is None else kv_type
    generator = torch.Generator(device="cuda").manual_seed(2026)

    def rand(dims, dtype):
        return (torch.rand(dims, device="cuda", generator=generator) * 0.2 - 0.1).to(dtype)

    # Larger Q/K magnitudes keep even long-row dQ above the absolute budget.
    q = (rand((tokens, heads, width), torch.float32) * 5).to(q_type)
    kv = (rand((keys, width), torch.float32) * 5).to(kv_type)
    sink = rand((heads,), torch.float32)
    # Repeated indices also exercise accumulation when topk > keys.
    indices = (torch.arange(topk, device="cuda", dtype=torch.int32) % keys)
    indices = indices.expand(tokens, -1).contiguous()
    lens = None
    if lengths:
        indices[:, -1] = keys + 1
        indices[:, -2] = -1
        lens = (torch.arange(tokens, device="cuda", dtype=torch.int32) % topk) + 1
        lens[0] = -1
        lens[-1] = topk + 3
    rq, rk, rs = [x.float().detach().requires_grad_() for x in (q, kv, sink)]
    output, lse = reference_forward(rq, rk, rs, indices, width**-0.5, value_dim, lens)
    out = output.detach().to(q_type)
    dout = rand(out.shape, q_type)
    expected = torch.autograd.grad(output, (rq, rk, rs), dout.float())
    args = (q, kv, out, dout, lse.detach(), sink, indices, width**-0.5, lens)
    return args, expected


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,value_dim,lengths",
    [
        ((8, 8, 512, 8, 128), 512, False),
        ((33, 8, 512, 17, 19), 512, True),
        ((128, 16, 512, 128, 128), 512, False),
        ((3, 65, 512, 9, 17), 512, True),
        ((3, 4, 576, 8, 17), 512, False),
        ((3, 8, 576, 9, 19), 576, True),
    ],
)
def test_matches_independent_autograd(dtype, shape, value_dim, lengths):
    args, expected = make_inputs(shape, dtype, value_dim=value_dim, lengths=lengths)
    actual = flash_train.torch.sparse_attn_bwd(*args)
    assert [t.dtype for t in actual] == [torch.bfloat16, torch.bfloat16, torch.float32]
    rtol, atol = TOLERANCES[dtype]
    for index, (result, reference) in enumerate(zip(actual, expected)):
        # Low-precision MMAC rounds P/dS before the shared dKV reduction.
        # Many-head cancellation needs a larger absolute dKV budget than dQ
        # and dSink; each output must still reject an all-zero implementation.
        abs_budget = atol * 10 if index == 1 and dtype != torch.float32 else atol
        assert torch.any(reference.abs() > abs_budget + rtol * reference.abs()), "oracle cannot reject zero output"
        torch.testing.assert_close(result.float(), reference, rtol=rtol, atol=abs_budget)
    if lengths:
        assert torch.count_nonzero(actual[0][0]) == 0


# Mixed-storage numerical cases are deferred with mixed-precision Primitives.
# Restore them in the Plan-enumerated C++ grid and Python API checks together.


def test_namespace_and_rank_three_kv():
    args, expected = make_inputs((3, 4, 512, 8, 17), torch.bfloat16)
    q, kv, out, dout, lse, sink, indices, scale, lengths = args
    result = flash_train.torch.DSA.sparse_attention_backward_wrapper(
        q, kv[:, None, :], out, dout, lse, sink, indices,
        softmax_scale=scale, topk_length=lengths,
    )
    assert set(result) == {"dq", "dkv", "d_sink"}
    assert result["dkv"].shape == (8, 1, 512)
    for actual, ref in zip(result.values(), expected):
        torch.testing.assert_close(actual.float().reshape(ref.shape), ref, rtol=5e-2, atol=2e-3)


def test_current_stream_and_graph_replay_reset_outputs():
    args, expected = make_inputs((3, 4, 512, 8, 17), torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = flash_train.torch.sparse_attn_bwd(*args)
    torch.cuda.current_stream().wait_stream(stream)
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result.float(), ref, rtol=5e-2, atol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flash_train.torch.sparse_attn_bwd(*args)
    incoming = args[3].clone()
    for multiplier in (1.0, -0.5, 1.0):
        args[3].copy_(incoming * multiplier)
        graph.replay()
        for result, ref in zip(actual, expected):
            torch.testing.assert_close(result.float(), ref * multiplier, rtol=5e-2, atol=2e-3)


def test_rejects_invalid_ports():
    args, _ = make_inputs((3, 4, 512, 8, 17), torch.bfloat16)
    for index, tensor in [(4, args[4].half()), (5, args[5].half()), (6, args[6].long())]:
        invalid = list(args)
        invalid[index] = tensor
        with pytest.raises(RuntimeError):
            flash_train.torch.sparse_attn_bwd(*invalid)
    for dtype in (torch.float16, torch.float32):
        invalid = tuple(x.to(dtype) if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 else x for x in args)
        with pytest.raises(RuntimeError, match="BF16"):
            flash_train.torch.sparse_attn_bwd(*invalid)
    invalid = list(args)
    invalid[3] = args[3][..., :-1].contiguous()
    with pytest.raises(RuntimeError):
        flash_train.torch.sparse_attn_bwd(*invalid)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two HCU devices")
def test_device_guard_restores_device_and_rejects_cross_device_inputs():
    args, expected = make_inputs((3, 4, 512, 8, 17), torch.bfloat16)
    original_device = torch.cuda.current_device()
    other_device = (original_device + 1) % torch.cuda.device_count()
    moved = tuple(x.to(other_device) if isinstance(x, torch.Tensor) else x for x in args)
    actual = flash_train.torch.sparse_attn_bwd(*moved)
    assert torch.cuda.current_device() == original_device
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result.float().cpu(), ref.cpu(), rtol=5e-2, atol=2e-3)
    invalid = list(moved)
    invalid[1] = args[1]
    with pytest.raises(RuntimeError, match="must share q's device"):
        flash_train.torch.sparse_attn_bwd(*invalid)
    assert torch.cuda.current_device() == original_device
