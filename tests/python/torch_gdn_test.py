# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from flash_train.torch import (
    chunk_gated_delta_rule, chunk_gdn2, gdn_forward, gdn_backward,
    gdn2_forward, gdn2_backward,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible HCU device")


def inputs(gdn2=False, s=512, h=1, hv=1, d=64):
    torch.manual_seed(17)
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, s, hv, d, device="cuda", dtype=torch.bfloat16) * 0.1
    if gdn2:
        g = torch.randn(1, s, hv, d, device="cuda").mul_(-0.02)
        b = torch.randn_like(g).mul_(0.02)
        w = torch.randn(1, s, hv, d, device="cuda").mul_(0.02)
        return q, k, v, g, b, w
    g = -torch.rand(1, s, hv, device="cuda") * 0.2
    beta = torch.sigmoid(torch.randn_like(g))
    return q, k, v, g, beta


@pytest.mark.parametrize("family", ["gdn", "gdn2"])
@pytest.mark.parametrize(
    "s,h,hv,d",
    [(512, 1, 1, 64), (512, 1, 2, 64), (1024, 16, 16, 128), (2048, 4, 16, 64)],
)
def test_low_level_and_autograd(family, s, h, hv, d):
    args = inputs(family == "gdn2", s=s, h=h, hv=hv, d=d)
    forward = gdn_forward if family == "gdn" else gdn2_forward
    backward = gdn_backward if family == "gdn" else gdn2_backward
    op = chunk_gated_delta_rule if family == "gdn" else chunk_gdn2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out, state = forward(*args)
        assert out.shape == args[2].shape and state is None
        grads = backward(*args, torch.ones_like(out))
        assert all(t.shape == src.shape for t, src in zip(grads[:3], args[:3]))
        tracked = tuple(t.detach().requires_grad_(True) for t in args)
        o, _ = op(*tracked)
        torch.testing.assert_close(o, out)
        o.sum().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tracked)
        assert tracked[-1].grad.dtype == tracked[-1].dtype
        for grad, low in zip((t.grad for t in tracked), grads):
            torch.testing.assert_close(grad, low.to(grad.dtype))
    stream.synchronize()


def test_gdn_state_and_checkpoint():
    q, k, v, g, beta = inputs()
    h0 = torch.zeros(1, 1, 64, 64, device="cuda", requires_grad=True)
    o, ht = chunk_gated_delta_rule(q.requires_grad_(), k, v, g, beta.to(torch.bfloat16), initial_state=h0,
                                    output_final_state=True)
    assert ht.shape == h0.shape
    (o.float().sum() + ht.sum()).backward()
    assert h0.grad is not None and h0.grad.shape == h0.shape
    assert q.grad is not None

    b = beta.detach().to(torch.bfloat16).requires_grad_()
    def run(q_, beta_):
        return chunk_gated_delta_rule(q_, k, v, g, beta_)[0]
    checkpoint(run, q.detach().requires_grad_(), b, use_reentrant=False).sum().backward()
    assert b.grad is not None and b.grad.dtype == b.dtype


def test_gdn2_checkpoint():
    args = tuple(t.detach().to(torch.bfloat16).requires_grad_(True) for t in inputs(True))
    def run(*values):
        return chunk_gdn2(*values)[0]
    checkpoint(run, *args, use_reentrant=False).sum().backward()
    assert all(t.grad is not None and t.grad.dtype == t.dtype for t in args)


def test_gdn_fp32_inference_only():
    q, k, v, g, beta = inputs(s=64)
    q, k, v = (t.float() for t in (q, k, v))
    o, ht = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    assert o.dtype == torch.float32 and ht.shape == (1, 1, 64, 64)
    assert torch.isfinite(o).all() and torch.isfinite(ht).all()
    with pytest.raises(ValueError, match="backward requires"):
        chunk_gated_delta_rule(q.requires_grad_(), k, v, g, beta)


@pytest.mark.parametrize("family", ["gdn", "gdn2"])
def test_unsupported(family):
    args = inputs(family == "gdn2", s=256)
    op = chunk_gated_delta_rule if family == "gdn" else chunk_gdn2
    with pytest.raises(ValueError, match="backward requires"):
        op(*tuple(t.detach().requires_grad_(True) for t in args))
    with pytest.raises(ValueError, match="chunk_size"):
        op(*args, chunk_size=32)
    with pytest.raises(TypeError, match="unsupported"):
        op(*args, made_up=True)
    with pytest.raises(ValueError, match="unsupported"):
        op(*args, cu_seqlens=torch.tensor([0, 256], device="cuda"))
    if family == "gdn2":
        with pytest.raises(ValueError, match="initial_state"):
            op(*args, initial_state=torch.zeros(1, 1, 64, 64, device="cuda"))
        with pytest.raises(ValueError, match="output_final_state"):
            op(*args, output_final_state=True)
    else:
        with pytest.raises(ValueError, match="use_gate_in_kernel"):
            op(*args, use_gate_in_kernel=True)


def test_low_level_rejects_inputs():
    q, k, v, g, beta = inputs()
    with pytest.raises(RuntimeError):
        gdn_forward(q.cpu(), k, v, g, beta)
    with pytest.raises(RuntimeError):
        gdn_backward(q, k, v, g, beta, v[:, :1].contiguous())
    with pytest.raises(RuntimeError):
        gdn2_forward(*inputs(True), scale=float("nan"))


def test_gdn_bf16_beta():
    # The low-level gdn_forward/gdn_backward bindings require beta to be fp32
    # (checkTensor(..., at::kFloat, ...) in gdn_binding.cpp); only the high-level
    # chunk_gated_delta_rule accepts bf16 beta and converts it internally before
    # calling the low-level function, returning dbeta in the caller's dtype.
    #
    # Tolerances are measured, not arbitrary: on this shape, true bf16-rounding
    # noise is max_abs~=3.1e-5 on the forward output (output max_abs~=7.5e-3,
    # ~0.4% relative) and max_abs~=3.1e-4 on dbeta (grad max_abs~=0.11, ~0.3%
    # relative). atol/rtol below give ~5-8x headroom over that noise floor while
    # still catching a gross conversion bug: a deliberately-scaled-by-0.5 beta
    # (a stand-in for "dropped a factor" class of bug) misses the loose
    # atol=0.05 this test used to have by ~13x, but blows past the tolerances
    # below by over an order of magnitude.
    q, k, v, g, beta = inputs()
    beta_bf16 = beta.detach().to(torch.bfloat16).requires_grad_(True)
    beta_fp32 = beta.detach().clone().requires_grad_(True)
    o_bf16, _ = chunk_gated_delta_rule(q, k, v, g, beta_bf16)
    o_fp32, _ = chunk_gated_delta_rule(q, k, v, g, beta_fp32)
    torch.testing.assert_close(o_bf16, o_fp32, atol=2e-5, rtol=0.02)
    o_bf16.sum().backward()
    o_fp32.sum().backward()
    assert beta_bf16.grad is not None and beta_bf16.grad.dtype == torch.bfloat16
    torch.testing.assert_close(beta_bf16.grad.float(), beta_fp32.grad.float(), atol=5e-5, rtol=0.02)


@pytest.mark.parametrize("field", ["g", "b", "w"])
def test_gdn2_bf16_gate(field):
    # Same contract as test_gdn_bf16_beta, but GDN2 allows bf16 on all three of
    # g/b/w (chunk_gdn2 converts each with .float() before the low-level call).
    # Measured noise floor for each field is max_abs<=8.6e-6 on the forward
    # output (output max_abs~=2.2e-3, ~0.4% relative) and max_abs<=9.8e-5 on
    # the gradient (grad max_abs ranges 5.3e-3..0.17, <=0.6% relative); the
    # same atol/rtol as the GDN case give comparable headroom here.
    q, k, v, g, b, w = inputs(True)
    tensors = {"g": g, "b": b, "w": w}
    bf16 = {
        name: (t.detach().to(torch.bfloat16).requires_grad_(True) if name == field
               else t.detach().requires_grad_(True))
        for name, t in tensors.items()
    }
    fp32 = {name: t.detach().clone().requires_grad_(True) for name, t in tensors.items()}
    o_bf16, _ = chunk_gdn2(q, k, v, bf16["g"], bf16["b"], bf16["w"])
    o_fp32, _ = chunk_gdn2(q, k, v, fp32["g"], fp32["b"], fp32["w"])
    torch.testing.assert_close(o_bf16, o_fp32, atol=2e-5, rtol=0.02)
    o_bf16.sum().backward()
    o_fp32.sum().backward()
    assert bf16[field].grad is not None and bf16[field].grad.dtype == torch.bfloat16
    torch.testing.assert_close(bf16[field].grad.float(), fp32[field].grad.float(), atol=5e-5, rtol=0.02)


def test_low_level_rejects_fp32_backward():
    # gdn_backward/gdn2_backward require bf16 q/k/v/do (forward tolerates fp32
    # via fp32_allowed, backward does not); called directly (bypassing the
    # autograd layer's own shape gate), fp32 inputs must raise RuntimeError.
    q, k, v, g, beta = inputs()
    with pytest.raises(RuntimeError):
        gdn_backward(q.float(), k.float(), v.float(), g, beta, v.float())
    q2, k2, v2, g2, b2, w2 = inputs(True)
    with pytest.raises(RuntimeError):
        gdn2_backward(q2.float(), k2.float(), v2.float(), g2, b2, w2, v2.float())


@pytest.mark.skip(
    reason="gfx938 rejection cannot be exercised from Python: the binding queries the real "
    "device architecture via hipGetDeviceProperties and does not expose the test-only "
    "Constraints(device_id, arch) constructor that tests/family/gdn/constraints_arch_test.cpp "
    "uses in C++, and this machine only has gfx936 hardware. Covered today only at the C++ "
    "layer. Add a real test here if a gfx938 machine becomes available or the binding grows "
    "a test-only architecture override."
)
def test_gdn_gfx938_rejected():
    pass
