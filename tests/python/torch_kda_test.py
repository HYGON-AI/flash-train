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
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith(("gfx936", "gfx938"))


pytestmark = pytest.mark.skipif(not _gfx936(), reason="the G2 KDA kernels target gfx936 / gfx938")

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
        ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"] - 10.0, x["beta"])


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

    def spy(*args, **kw):
        seen.append(args[-3:])  # keep_state, save_qkn, save_h
        return real(*args, **kw)
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

    def spy(*args, **kw):
        seen.append(args[-3:])  # keep_state, save_qkn, save_h
        return real(*args, **kw)
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


_KERNELS_SCRIPT = r"""
import torch, flash_train.torch as ft
from torch.profiler import ProfilerActivity, profile
T, H = 256, 4
x = [(torch.randn(T, H, 128, device="cuda") * .3).bfloat16().requires_grad_() for _ in range(4)]
beta = torch.rand(T, H, device="cuda").requires_grad_()
a, d = torch.zeros(H, device="cuda"), torch.zeros(H * 128, device="cuda")
with profile(activities=[ProfilerActivity.CUDA]) as p:
    ft.kimi_delta_attention(*x, beta, A_log=a, dt_bias=d).sum().backward()
    torch.cuda.synchronize()
print(sorted({e.key for e in p.key_averages() if "g2" in e.key}))
"""


def _kernel_names(**env):
    import subprocess
    import sys
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("KDA_G2_", "FTRAIN_KDA_DEV"))}
    out = subprocess.run([sys.executable, "-c", _KERNELS_SCRIPT], env={**clean, **env},
                         capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]


def test_kernel_knobs_need_ftrain_kda_dev():
    """The G2 kernels' KDA_G2_* knobs (hip_kda research switches) are ignored unless FTRAIN_KDA_DEV
    is set: a hip_kda environment cannot change flash-train's kernel choice (each process reads
    the switch once, hence the subprocesses)."""
    base = _kernel_names()
    knob = {"KDA_G2_S4A_AFS_MAXH": "0"}  # H = 4 <= 12 takes the AFS s4a instance by default
    assert _kernel_names(**knob) == base
    assert _kernel_names(**knob, FTRAIN_KDA_DEV="0") == base
    assert _kernel_names(**knob, FTRAIN_KDA_DEV="1") != base


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


def test_chunk_kda_passes_the_states_through():
    """chunk_kda's initial_state / output_final_state (FLA's [N, H, K, V], or [N, H, V, K] with
    transpose_state_layout) are kimi_delta_attention's."""
    x = _fla_inputs(1, 256, 2, seed=7)
    h0 = torch.randn(1, 2, 128, 128, device="cuda") * 0.1
    k3 = dict(A_log=x["A_log"], dt_bias=x["dt_bias"], use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              safe_gate=True, lower_bound=-5.0)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    thd = [t[0] for t in args]
    ro, rht = ft.kimi_delta_attention(*thd, A_log=x["A_log"], dt_bias=x["dt_bias"], initial_state=h0,
                                      output_final_state=True)
    o, ht = ft.chunk_kda(*args, initial_state=h0, output_final_state=True, **k3)
    assert torch.equal(o[0], ro) and torch.equal(ht, rht)
    o, ht = ft.chunk_kda(*args, initial_state=h0.transpose(-1, -2), output_final_state=True,
                         transpose_state_layout=True, **k3)
    assert torch.equal(o[0], ro) and torch.equal(ht, rht.transpose(-1, -2))


def test_chunk_kda_rejects_unsupported_arguments():
    x = _fla_inputs(1, 64, 2, seed=1)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    k3 = dict(A_log=x["A_log"], dt_bias=x["dt_bias"], use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              safe_gate=True, lower_bound=-5.0)
    with pytest.raises(ValueError):
        ft.chunk_kda(*args, output_final_state=True, cp_context=object(), **k3)
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


def make_h0(N, H, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(N, H, 128, 128, device="cuda", generator=gen) * 0.1


def call(fn, x, **kw):
    return fn(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x.get("A_log"), dt_bias=x.get("dt_bias"), **kw)


@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_state_ports_match_fp64_reference(T, H, cu, gate_in_kernel):
    x = make_inputs(T, H, gate_in_kernel, seed=T + H + 7)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    h0 = make_h0(1 if cu is None else len(cu) - 1, H, seed=T + H + 8)
    with torch.no_grad():
        o, ht = call(ft.kimi_delta_attention, x, cu_seqlens=cu_t, initial_state=h0, output_final_state=True)
        ro, rht = call(ft.kda_reference, x, cu_seqlens=cu_t, initial_state=h0, output_final_state=True)
    assert ht.dtype == torch.float32 and ht.shape == h0.shape
    assert torch.isfinite(o.float()).all() and torch.isfinite(ht).all()
    assert rel_rms(o, ro) < TOL["o"], rel_rms(o, ro)
    assert rel_rms(ht, rht) < TOL["o"], rel_rms(ht, rht)


def test_zero_initial_state_is_bit_identical():
    x = make_inputs(333, 2, True, seed=21)
    cu = torch.tensor([0, 1, 64, 65, 200, 333], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        o = call(ft.kimi_delta_attention, x, cu_seqlens=cu)
        o0, _ = call(ft.kimi_delta_attention, x, cu_seqlens=cu, initial_state=torch.zeros(5, 2, 128, 128,
                     device="cuda"), output_final_state=True)
    assert torch.equal(o, o0)


@pytest.mark.parametrize("gate_in_kernel", [True, False])
def test_chunk_aligned_split_chains_the_state(gate_in_kernel):
    """What chunkwise CP relies on: running [0, s) then [s, T) seeded with the first part's final
    state is bit-identical to one call over [0, T) when s is a multiple of the chunk (64)."""
    T, H, s = 512, 4, 192
    x = make_inputs(T, H, gate_in_kernel, seed=31)
    part = lambda a, b: {k: (t[a:b] if k not in ("A_log", "dt_bias") else t) for k, t in x.items()}
    with torch.no_grad():
        o, ht = call(ft.kimi_delta_attention, x, output_final_state=True)
        o1, h1 = call(ft.kimi_delta_attention, part(0, s), output_final_state=True)
        o2, h2 = call(ft.kimi_delta_attention, part(s, T), initial_state=h1, output_final_state=True)
    assert torch.equal(torch.cat([o1, o2]), o)
    assert torch.equal(h2, ht)


def test_empty_sequence_passes_the_initial_state_through():
    x = make_inputs(200, 3, True, seed=41)
    cu = torch.tensor([0, 0, 63, 127, 127, 200], dtype=torch.int32, device="cuda")
    h0 = make_h0(5, 3, seed=42)
    with torch.no_grad():
        _, ht = call(ft.kimi_delta_attention, x, cu_seqlens=cu, initial_state=h0, output_final_state=True)
    assert torch.equal(ht[0], h0[0]) and torch.equal(ht[3], h0[3])


def run_state(fn, x, h0, d_o, d_ht, **kw):
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
    leaves["h0"] = h0.detach().clone().requires_grad_(True)
    o, ht = fn(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], A_log=leaves.get("A_log"),
               dt_bias=leaves.get("dt_bias"), initial_state=leaves["h0"], output_final_state=True, **kw)
    grads = torch.autograd.grad((o, ht), list(leaves.values()), (d_o.to(o.dtype), d_ht))
    return o.detach(), ht.detach(), dict(zip(leaves, grads))


@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_state_gradients_match_fp64_reference(T, H, cu, gate_in_kernel):
    """With an initial_state and a final-state gradient dht: dh0 and every input gradient."""
    x = make_inputs(T, H, gate_in_kernel, seed=T + H + 11)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    N = 1 if cu is None else len(cu) - 1
    h0, d_ht = make_h0(N, H, seed=T + H + 12), make_h0(N, H, seed=T + H + 13)
    d_o = make_d_o(T, H, seed=T + H + 14)
    _, _, g = run_state(ft.kimi_delta_attention, x, h0, d_o, d_ht, cu_seqlens=cu_t)
    _, _, rg = run_state(ft.kda_reference, x, h0, d_o, d_ht, cu_seqlens=cu_t)
    for name in x:
        assert rel_rms(g[name], rg[name]) < TOL[name], name
    assert g["h0"].dtype == torch.float32 and g["h0"].shape == h0.shape
    assert rel_rms(g["h0"], rg["h0"]) < 4e-2


@pytest.mark.parametrize("T,H,cu", [(257, 4, None), (200, 3, [0, 0, 63, 127, 127, 200])])
def test_state_keep_state_is_bit_identical(T, H, cu):
    x = make_inputs(T, H, True, seed=T + 21)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    N = 1 if cu is None else len(cu) - 1
    args = (x, make_h0(N, H, seed=T + 22), make_d_o(T, H, seed=T + 23), make_h0(N, H, seed=T + 24))
    a = _with_keep("0", lambda: run_state(ft.kimi_delta_attention, *args, cu_seqlens=cu_t))
    b = _with_keep("1", lambda: run_state(ft.kimi_delta_attention, *args, cu_seqlens=cu_t))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for name in a[2]:
        assert torch.equal(a[2][name], b[2][name]), name


@pytest.mark.parametrize("gate_in_kernel", [True, False])
def test_chunk_aligned_split_chains_the_gradients(gate_in_kernel):
    """The backward chunkwise CP relies on: the second part's dh0 seeds the first part's dht, and
    every gradient equals the one-call gradient over [0, T) bit for bit, except the per-call
    reductions: dA_log adds two fp32 partial sums, ddt_bias two bf16-rounded ones (FLA's contract)."""
    T, H, s = 512, 4, 192
    x = make_inputs(T, H, gate_in_kernel, seed=33)
    d_o, d_ht = make_d_o(T, H, seed=34), make_h0(1, H, seed=35)
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}

    def fwd(a, b, **kw):
        part = {n: (t[a:b] if n not in ("A_log", "dt_bias") else t) for n, t in leaves.items()}
        return call(ft.kimi_delta_attention, part, output_final_state=True, **kw)

    o, ht = fwd(0, T)
    full = torch.autograd.grad((o, ht), list(leaves.values()), (d_o, d_ht))
    o1, h1 = fwd(0, s)
    o2, h2 = fwd(s, T, initial_state=h1)
    split = torch.autograd.grad((o1, o2, h2), list(leaves.values()), (d_o[:s], d_o[s:], d_ht))
    for name, a, b in zip(leaves, full, split):
        if name in ("A_log", "dt_bias"):
            assert rel_rms(b, a) < (1e-5 if name == "A_log" else 1e-2), name
        else:
            assert torch.equal(a, b), name
