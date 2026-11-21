# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash_train.torch.kimi_delta_attention against its PyTorch FP64 reference.

Tolerances (relative RMS error vs the fp64 reference, the cuDNN-frontend PR #556 criteria):
o 2e-2; dq, dk, dv 4e-2; dg, dbeta, dA_log, ddt_bias 6e-2. The G2 kernels compute in bf16
operands with fp32 accumulation; observed errors are ~0.3-0.7%.
"""

import os

import pytest
import torch

import flash_train.torch as ft


def _gfx936():
    if not torch.cuda.is_available():
        return False
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith("gfx936")


pytestmark = pytest.mark.skipif(not _gfx936(), reason="the G2 KDA kernels target gfx936")

TOL = {"o": 2e-2, "q": 4e-2, "k": 4e-2, "v": 4e-2, "g": 6e-2, "beta": 6e-2, "A_log": 6e-2, "dt_bias": 6e-2}


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-6)).item()


def make_inputs(T, H, gate_in_kernel, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    x = {"q": (r(T, H, 128) * 0.3).bfloat16(), "k": (r(T, H, 128) * 0.3).bfloat16(),
         "v": (r(T, H, 128) * 0.3).bfloat16(), "beta": r(T, H).sigmoid()}
    if gate_in_kernel:
        x["g"] = (r(T, H, 128) * 0.3).bfloat16()
        x["A_log"] = torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen))
        x["dt_bias"] = r(H * 128) * 0.1
    else:
        x["g"] = -torch.rand(T, H, 128, device="cuda", generator=gen) * 0.5
    return x


def make_d_o(T, H, seed):
    # seeded: the default CUDA generator's seed differs per process, which made the
    # tolerance checks (e.g. the 2-element dA_log of H = 2) vary from run to run
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(T, H, 128, device="cuda", generator=gen).bfloat16()


def run(fn, x, d_o, **kw):
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
    o = fn(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], A_log=leaves.get("A_log"),
           dt_bias=leaves.get("dt_bias"), **kw)
    grads = torch.autograd.grad(o, list(leaves.values()), d_o.to(o.dtype))
    return o.detach(), dict(zip(leaves, grads))


CASES = [(64, 2, None), (100, 2, None), (257, 4, None), (512, 4, None),
         (333, 2, [0, 1, 64, 65, 200, 333]), (200, 3, [0, 0, 63, 127, 127, 200])]


@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_matches_fp64_reference(T, H, cu, gate_in_kernel):
    x = make_inputs(T, H, gate_in_kernel, seed=T + H)
    d_o = make_d_o(T, H, seed=T + H + 1)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    o, g = run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t)
    ro, rg = run(ft.kda_reference, x, d_o, cu_seqlens=cu_t)
    assert torch.isfinite(o.float()).all()
    assert rel_rms(o, ro) < TOL["o"]
    for name in x:
        assert torch.isfinite(g[name].float()).all(), name
        assert rel_rms(g[name], rg[name]) < TOL[name], (name, rel_rms(g[name], rg[name]))


def test_deterministic():
    x = make_inputs(512, 4, True, seed=3)
    d_o = make_d_o(512, 4, seed=4)
    a = run(ft.kimi_delta_attention, x, d_o)
    b = run(ft.kimi_delta_attention, x, d_o)
    assert torch.equal(a[0], b[0])
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def test_packed_equals_separate():
    """Sequences in one packed call are independent: bit-identical to separate varlen calls,
    and to the dense single-sequence path (no cu_seqlens, T % 64 == 0)."""
    lengths = [128, 64, 192]
    T, H = sum(lengths), 2
    x = make_inputs(T, H, True, seed=11)
    d_o = make_d_o(T, H, seed=12)
    cu = torch.tensor([0, 128, 192, 384], dtype=torch.int32, device="cuda")
    o, g = run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu)
    s = 0
    for n in lengths:
        part = {k: (t[s:s + n] if k not in ("A_log", "dt_bias") else t) for k, t in x.items()}
        one = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        op, gp = run(ft.kimi_delta_attention, part, d_o[s:s + n], cu_seqlens=one)
        assert torch.equal(op, o[s:s + n])
        for name in ("q", "k", "v", "g", "beta"):
            assert torch.equal(gp[name], g[name][s:s + n]), name
        od, gd = run(ft.kimi_delta_attention, part, d_o[s:s + n])
        assert torch.equal(od, op)
        for name in ("q", "k", "v", "g", "beta"):
            assert torch.equal(gd[name], gp[name]), name
        s += n


def test_rejects_unsupported_head_dim():
    x = make_inputs(64, 2, True)
    x = {k: (t[..., :64].contiguous() if k in ("q", "k", "v", "g") else t) for k, t in x.items()}
    x["dt_bias"] = x["dt_bias"][: 2 * 64]
    with pytest.raises(RuntimeError):
        ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x["A_log"], dt_bias=x["dt_bias"])


def test_rejects_decay_below_bound():
    x = make_inputs(64, 2, False)
    with pytest.raises(ValueError):
        ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"] - 10.0, x["beta"], check_decay=True)


def _with_env(env, fn):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v


def _with_keep(mode, fn):
    # keep-state only: saved-h (the default at H <= 64) would take its place
    return _with_env({"FTRAIN_KDA_KEEP_STATE": mode, "FTRAIN_KDA_SAVE_H": "0"}, fn)


# What the forward saves for the backward: recompute everything / keep-state / saved-h, each
# with q, k or with the saved qn, kn. Every route is bit-identical to the full recompute.
ROUTES = {
    "keep": {"FTRAIN_KDA_KEEP_STATE": "1", "FTRAIN_KDA_SAVE_H": "0", "FTRAIN_KDA_SAVE_QKN": "0"},
    "keep+qkn": {"FTRAIN_KDA_KEEP_STATE": "1", "FTRAIN_KDA_SAVE_H": "0", "FTRAIN_KDA_SAVE_QKN": "1"},
    "h": {"FTRAIN_KDA_KEEP_STATE": "0", "FTRAIN_KDA_SAVE_H": "1", "FTRAIN_KDA_SAVE_QKN": "0"},
    "h+qkn": {"FTRAIN_KDA_KEEP_STATE": "0", "FTRAIN_KDA_SAVE_H": "1", "FTRAIN_KDA_SAVE_QKN": "1"},
    "qkn": {"FTRAIN_KDA_KEEP_STATE": "0", "FTRAIN_KDA_SAVE_H": "0", "FTRAIN_KDA_SAVE_QKN": "1"},
}
RECOMPUTE = {"FTRAIN_KDA_KEEP_STATE": "0", "FTRAIN_KDA_SAVE_H": "0", "FTRAIN_KDA_SAVE_QKN": "0"}


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_saved_routes_are_bit_identical(T, H, cu, gate_in_kernel, route):
    """The forward handing qn / kn, h or its full recurrent state to the backward (instead of
    the backward recomputing them) changes nothing: o and every gradient bit for bit."""
    x = make_inputs(T, H, gate_in_kernel, seed=2 * T + H)
    d_o = make_d_o(T, H, seed=2 * T + H + 1)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    a = _with_env(RECOMPUTE, lambda: run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t))
    b = _with_env(ROUTES[route], lambda: run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t))
    assert torch.equal(a[0], b[0])
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def test_saved_h_skips_the_backward_recurrence():
    """Saved-h (the default at H <= 64): a plain training step launches the recurrence once
    (forward only); with it off the backward recomputes it."""
    T, H = 512, 4
    x = make_inputs(T, H, True, seed=7)
    d_o = make_d_o(T, H, seed=8)
    step = lambda: run(ft.kimi_delta_attention, x, d_o)
    assert _with_env({"FTRAIN_KDA_SAVE_H": "auto"}, lambda: _recurrence_launches(step)) == 1
    assert _with_env({"FTRAIN_KDA_SAVE_H": "0", "FTRAIN_KDA_KEEP_STATE": "0"},
                     lambda: _recurrence_launches(step)) == 2


def test_no_grad_forward_saves_nothing(monkeypatch):
    """A forward under no_grad (inference, an activation checkpoint's first pass) writes none of
    the optional backward outputs, although needs_input_grad stays True there."""
    import flash_train.torch.kda as kda_mod
    seen = []
    real = kda_mod._kda_forward

    def spy(*args):
        seen.append(args[-3:])  # keep_state, save_qkn, save_h
        return real(*args)
    monkeypatch.setattr(kda_mod, "_kda_forward", spy)
    x = {n: t.requires_grad_(True) for n, t in make_inputs(128, 2, True, seed=9).items()}
    call = lambda: ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x["A_log"],
                                           dt_bias=x["dt_bias"])
    with torch.no_grad():
        a = call()
    b = call()
    assert seen == [(False, False, False), (False, True, True)]
    assert torch.equal(a, b.detach())


def _spy_saves(monkeypatch):
    import flash_train.torch.kda as kda_mod
    seen = []
    real = kda_mod._kda_forward

    def spy(*args):
        seen.append(args[-3:])  # keep_state, save_qkn, save_h
        return real(*args)
    monkeypatch.setattr(kda_mod, "_kda_forward", spy)
    return seen


@pytest.mark.parametrize("H,policy,want", [
    (4, None, (False, True, True)),    # measured policy: save h at <= 64 heads
    (4, True, (False, True, True)),
    (4, False, (False, True, False)),  # minimal set: no h
    (64, None, (False, True, True)),
    (68, None, (False, True, False)),  # > 64 heads: no h outside a checkpoint recompute
    (68, True, (False, True, True)),
])
def test_disable_recompute_selects_the_saved_set(monkeypatch, H, policy, want):
    """chunk_kda's disable_recompute (FLA's meaning) picks what the forward keeps; the result is
    bit-identical to the measured policy either way."""
    seen = _spy_saves(monkeypatch)
    x = _fla_inputs(1, 128, H, seed=H)
    d_o = make_d_o(128, H, seed=H + 1).view(1, 128, H, 128)
    o, g = _k3_call(ft.chunk_kda, x, d_o, disable_recompute=policy)
    ro, rg = _k3_call(ft.chunk_kda, x, d_o)
    assert seen[0] == want
    assert torch.equal(o, ro)
    for name in x:
        assert torch.equal(g[name], rg[name]), name


def test_explicit_policy_overrides_the_debug_variables(monkeypatch):
    seen = _spy_saves(monkeypatch)
    x = _fla_inputs(1, 128, 4, seed=3)
    d_o = make_d_o(128, 4, seed=4).view(1, 128, 4, 128)
    _with_env({"FTRAIN_KDA_SAVE_H": "0"}, lambda: _k3_call(ft.chunk_kda, x, d_o))
    _with_env({"FTRAIN_KDA_SAVE_H": "0"}, lambda: _k3_call(ft.chunk_kda, x, d_o, disable_recompute=True))
    assert [s[2] for s in seen] == [False, True]


def test_dq_dk_buffer_hosts_the_gate_scratch():
    """With qn / kn saved, dq | dk come back as one buffer that hosted the gate scratch; the
    gradients equal the q / k route bit for bit."""
    T, H = 256, 2
    x = make_inputs(T, H, True, seed=10)
    d_o = make_d_o(T, H, seed=11)
    a = _with_env(ROUTES["qkn"], lambda: run(ft.kimi_delta_attention, x, d_o))
    b = _with_env(RECOMPUTE, lambda: run(ft.kimi_delta_attention, x, d_o))
    dq, dk = a[1]["q"], a[1]["k"]
    assert dk.data_ptr() == dq.data_ptr() + dq.numel() * dq.element_size()
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def _fla_inputs(B, T, H, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    return {"q": (r(B, T, H, 128) * 0.3).bfloat16(), "k": (r(B, T, H, 128) * 0.3).bfloat16(),
            "v": (r(B, T, H, 128) * 0.3).bfloat16(), "g": (r(B, T, H, 128) * 0.3).bfloat16(),
            "beta": r(B, T, H).sigmoid(),
            "A_log": torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen)),
            "dt_bias": r(H * 128) * 0.1}


def _k3_call(fn, x, d_o, **kw):
    """Megatron-Bridge's Kimi K3 call (kimi_k3_ops.kda -> chunk_kda), arguments as it passes them."""
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
    o, final_state = fn(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                        A_log=leaves["A_log"], dt_bias=leaves["dt_bias"], initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                        safe_gate=True, lower_bound=-5.0, transpose_state_layout=True, **kw)
    assert final_state is None
    grads = torch.autograd.grad(o, list(leaves.values()), d_o)
    return o.detach(), dict(zip(leaves, grads))


@pytest.mark.parametrize("B,T,H,cu", [(1, 512, 4, None), (2, 256, 4, None), (1, 384, 2, [0, 128, 192, 384])])
def test_chunk_kda_matches_kimi_delta_attention(B, T, H, cu):
    """flash_train.torch.chunk_kda (FLA's chunk_kda signature, [B, T, H, D]) is a pure layout
    adapter: bit-identical to kimi_delta_attention on the packed THD tensors."""
    x = _fla_inputs(B, T, H, seed=B * T + H)
    d_o = make_d_o(B * T, H, seed=B * T + H + 1).view(B, T, H, 128)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    o, g = _k3_call(ft.chunk_kda, x, d_o, cu_seqlens=cu_t)
    assert o.shape == (B, T, H, 128)
    if B > 1:
        cu_t = torch.arange(0, B * T + 1, T, dtype=torch.int32, device="cuda")
    thd = {n: (t.reshape(B * T, *t.shape[2:]) if n not in ("A_log", "dt_bias") else t) for n, t in x.items()}
    ro, rg = run(ft.kimi_delta_attention, thd, d_o.reshape(B * T, H, 128), cu_seqlens=cu_t)
    assert torch.equal(o.reshape(B * T, H, 128), ro)
    for name in x:
        assert torch.equal(g[name].reshape(rg[name].shape), rg[name]), name


def test_chunk_kda_rejects_unsupported_arguments():
    x = _fla_inputs(1, 64, 2, seed=1)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    k3 = dict(A_log=x["A_log"], dt_bias=x["dt_bias"], use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              safe_gate=True, lower_bound=-5.0)
    with pytest.raises(NotImplementedError):
        ft.chunk_kda(*args, output_final_state=True, **k3)
    with pytest.raises(NotImplementedError):
        ft.chunk_kda(*args, initial_state=torch.zeros(1, 2, 128, 128, device="cuda"), **k3)
    with pytest.raises(NotImplementedError):  # FLA's softplus gate (no lower_bound)
        ft.chunk_kda(*args, A_log=x["A_log"], dt_bias=x["dt_bias"], use_gate_in_kernel=True)
    with pytest.raises(TypeError):
        ft.chunk_kda(*args, not_an_fla_argument=1, **k3)


def _recurrence_launches(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(e.count for e in p.key_averages() if "g2_fwd_h_v8o" in e.key)


def test_checkpoint_recompute_keeps_state():
    """Under a reentrant activation checkpoint (Megatron's recompute form) the recompute forward
    keeps the state (auto): the backward runs no recurrence of its own -- 2 recurrence launches
    (no-grad forward + recompute) instead of 3 -- and the gradients equal a plain call bit for bit."""
    import torch.utils.checkpoint as tuc
    T, H = 512, 4
    x = make_inputs(T, H, True, seed=5)
    d_o = make_d_o(T, H, seed=6)
    plain = run(ft.kimi_delta_attention, x, d_o)

    def ckpt(mode):
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
        order = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
        f = lambda q, k, v, g, beta, a, dt: ft.kimi_delta_attention(q, k, v, g, beta, A_log=a, dt_bias=dt)

        def step():
            o = tuc.checkpoint(f, *(leaves[n] for n in order), use_reentrant=True)
            o.backward(d_o)
            return o
        launches = _with_keep(mode, lambda: _recurrence_launches(step))
        return launches, leaves

    for mode, want in (("auto", 2), ("0", 3)):
        launches, leaves = ckpt(mode)
        assert launches == want, (mode, launches)
        for name in x:
            assert torch.equal(leaves[name].grad, plain[1][name]), (mode, name)
