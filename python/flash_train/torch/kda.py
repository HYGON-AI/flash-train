# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention (KDA): flash-train operator plus its PyTorch reference.

Layout is packed THD: q, k [T, H, K], v [T, H, V], g [T, H, K], beta [T, H]; ``cu_seqlens``
(int32 [N + 1]) splits the T rows into independent sequences (every sequence starts from a
zero state). Per head, with state S [K, V] and token t:

    S <- diag(exp(g_t)) S                      (per-key-channel decay)
    S <- S + beta_t k_t (v_t - S^T k_t)^T       (delta rule on the decayed state)
    o_t = S^T (scale * q_t)

Gate contracts:
  * ``A_log``/``dt_bias`` given (Kimi K3): g is the raw gate and the operator uses
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` as the log decay;
  * otherwise (cuDNN-frontend PR #556): g is the per-token log decay itself.
``use_qk_l2norm_in_kernel`` l2-normalizes q and k (eps 1e-6) before the recurrence.
"""

import math
import os

import torch

from ._flash_train_torch import kda_backward as _kda_backward
from ._flash_train_torch import kda_forward as _kda_forward

# The chunk algorithm's safe-gate range: the per-token log decay must stay >= this.
DECAY_LOWER_BOUND = -5.0

# Saved-h (the forward saves the recurrent state h, 512 B per token and head, and the backward runs
# no recurrence) is the default up to this many heads per call: measured faster than both the
# backward recompute and the keep-state hand-over on gfx936 (H = 8 .. 64; at H = 64, Megatron-LM-k3
# TP1, 1.03-1.08x over T 1K-32K); at H = 96 the forward's extra h store does not pay and the backward
# recomputes (or takes the keep-state hand-over).
SAVE_H_MAX_HEADS = 64


def _thd(x, name):
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(f"{name}: use packed THD with cu_seqlens instead of B > 1")
        x = x.squeeze(0)
    return x.contiguous()


def _keep_state(needs_bwd):
    """Should the forward hand its recurrent state to the backward (skipping the backward's
    recompute of it; bit-identical either way)?

    FTRAIN_KDA_KEEP_STATE=auto (default): only when this forward runs inside a backward pass
    with no saved-tensor hooks active -- a reentrant activation-checkpoint recompute (Megatron's
    CheckpointFunction, torch.utils.checkpoint(use_reentrant=True)) -- so the state lives just
    for this layer's backward. A plain training forward, non-reentrant checkpoint (its backward
    uses the original forward's ctx) or activation offloading keep the minimal saved set.
    1: whenever a backward follows; 0: never."""
    mode = os.environ.get("FTRAIN_KDA_KEEP_STATE", "auto").strip()
    if mode == "0" or not needs_bwd:
        return False
    if mode == "1":
        return True
    return (torch._C._current_graph_task_id() != -1
            and torch._C._autograd._top_saved_tensors_default_hooks(False) is None)


def _save_h(needs_bwd, heads):
    """Should the forward save its recurrent state h (512 B per token and head) so the backward
    runs no recurrence (bit-identical either way)?

    FTRAIN_KDA_SAVE_H=auto (default): whenever a backward follows and heads <= SAVE_H_MAX_HEADS;
    it then also replaces the keep-state hand-over. 1: whenever a backward follows; 0: never."""
    mode = os.environ.get("FTRAIN_KDA_SAVE_H", "auto").strip()
    if mode == "0" or not needs_bwd:
        return False
    return mode == "1" or heads <= SAVE_H_MAX_HEADS


def _save_qkn(needs_bwd, l2norm):
    """Should the forward save the l2-normalized qn / kn instead of q / k (the backward then skips
    rebuilding them; same bytes saved)? FTRAIN_KDA_SAVE_QKN=1 (default) / 0."""
    return (l2norm and needs_bwd
            and os.environ.get("FTRAIN_KDA_SAVE_QKN", "1").strip() != "0")


class _KdaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, scale, lower_bound, l2norm, grad_mode,
                disable_recompute, initial_state, output_final_state):
        # A backward follows only when the caller ran with grad mode on (inside forward it is always
        # off, and needs_input_grad stays True under no_grad, e.g. a checkpoint's first forward).
        needs_bwd = grad_mode and any(ctx.needs_input_grad)
        gate = a_log is not None
        if disable_recompute is None:   # the measured policy (FTRAIN_KDA_* in env.h override it)
            save_h = _save_h(needs_bwd, q.shape[1])
            save_qkn = _save_qkn(needs_bwd, l2norm)
            keep = _keep_state(needs_bwd) and not save_h
        else:   # the caller's choice: True saves h (no backward recurrence), False the minimal set
            save_h, save_qkn, keep = needs_bwd and disable_recompute, needs_bwd and l2norm, False
        save_qkn = save_qkn and q.numel() > 0
        o, q_rstd, k_rstd, aqk, akk, qn, kn, h, state, final_state, _ws = _kda_forward(
            q, k, v, g, beta, a_log, dt_bias, cu_seqlens, scale, lower_bound, gate, l2norm, keep, save_qkn, save_h,
            initial_state=initial_state, output_final_state=output_final_state)
        q_sv, k_sv = (None, None) if save_qkn else (q, k)   # qn / kn replace q / k in the saved set
        ctx.save_for_backward(q_sv, k_sv, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk,
                              qn, kn, h, state, initial_state)
        ctx.scale, ctx.lower_bound, ctx.gate, ctx.l2norm = scale, lower_bound, gate, l2norm
        ctx.set_materialize_grads(False)
        return o, (final_state if output_final_state else None)

    @staticmethod
    def backward(ctx, d_o, d_ht):
        (q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, h, state,
         initial_state) = ctx.saved_tensors
        want_dh0 = ctx.needs_input_grad[13]
        if d_o is None:
            d_o = torch.zeros_like(v)
        dht = d_ht.float().contiguous() if d_ht is not None else None
        dq, dk, dv, dg, dbeta, da_log, ddt_bias, dh0, *_cp = _kda_backward(
            d_o.contiguous(), q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, h,
            state, ctx.scale, ctx.lower_bound, ctx.gate, ctx.l2norm, initial_state, dht, want_dh0)
        if not ctx.gate:
            da_log = ddt_bias = None
        return (dq, dk, dv, dg, dbeta, da_log, ddt_bias, None, None, None, None, None, None,
                dh0 if want_dh0 else None, None)


def kimi_delta_attention(q, k, v, g, beta, *, cu_seqlens=None, scale=None, A_log=None, dt_bias=None,
                         lower_bound=-5.0, use_qk_l2norm_in_kernel=True, check_decay=True, disable_recompute=None,
                         initial_state=None, output_final_state=False):
    """KDA forward (differentiable) on the current stream; returns o [T, H, V] (bf16), or
    (o, final_state) with ``output_final_state``.

    ``initial_state``: fp32 [N, H, 128, 128] (key-major, FLA's [N, H, K, V]; N = 1 without
    cu_seqlens), each sequence's start state instead of zero. ``final_state`` (same layout) is
    each sequence's state after its last token. Both are differentiable: the backward takes the
    final state's gradient (dht) and returns the initial state's (dh0).

    q, k, v: bf16 [T, H, 128]; beta: post-activation [T, H] (computed in fp32);
    g: bf16 raw gate with ``A_log`` (fp32-able [H]) / ``dt_bias`` ([H, 128] or [H*128]),
    else the per-token log decay (fp32). ``cu_seqlens``: int32 [N + 1] on the device.
    ``check_decay`` (precomputed-g contract only) verifies g >= -5, the range the chunk
    algorithm requires (one device reduction and a host read per call).
    ``disable_recompute`` (FLA's meaning) chooses what the forward keeps for the backward; every
    choice is bit-identical. None (default): the measured policy -- save the recurrent state h
    (512 B per token and head) at <= 64 heads, hand the full state over inside a reentrant
    activation-checkpoint recompute; True: always save h, the backward runs no recurrence;
    False: the minimal set (FLA's default memory), the backward recomputes the recurrence.
    Raises RuntimeError (FTRAIN_STATUS_UNSUPPORTED) for configurations no flash-train
    implementation supports: the G2 kernels run on gfx936 / gfx938 with K = V = 128.
    """
    q, k, v, g = (_thd(t, n) for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g")))
    beta_in = _thd(beta.unsqueeze(-1), "beta").squeeze(-1) if beta.dim() == 3 else beta.contiguous()
    beta32 = beta_in.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if cu_seqlens is not None:
        # One sequence ([0, T]) is the dense layout: bit-identical, and the dense kernel instances
        # are faster than the packed ones (no host read of cu_seqlens needed to tell).
        cu_seqlens = None if cu_seqlens.numel() == 2 else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    if initial_state is not None:
        initial_state = initial_state.to(device=q.device, dtype=torch.float32).contiguous()
    if A_log is not None:
        if dt_bias is None:
            raise ValueError("kimi_delta_attention: A_log requires dt_bias")
        a32, d32 = A_log.float().contiguous(), dt_bias.float().contiguous()
        o, ht = _KdaFunction.apply(q, k, v, g.to(torch.bfloat16), beta32, a32, d32, cu_seqlens, float(scale),
                                   float(lower_bound), bool(use_qk_l2norm_in_kernel), torch.is_grad_enabled(),
                                   disable_recompute, initial_state, bool(output_final_state))
    else:
        g32 = g.float()
        if check_decay and g32.numel() and float(g32.detach().amin()) < DECAY_LOWER_BOUND:
            raise ValueError("kimi_delta_attention: per-token log decay below -5 is outside the supported range")
        o, ht = _KdaFunction.apply(q, k, v, g32, beta32, None, None, cu_seqlens, float(scale), 0.0,
                                   bool(use_qk_l2norm_in_kernel), torch.is_grad_enabled(), disable_recompute,
                                   initial_state, bool(output_final_state))
    return (o, ht) if output_final_state else o


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              allow_neg_eigval=False, safe_gate=False, lower_bound=None, disable_recompute=None,
              return_intermediate_states=False, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
              cp_context=None, chunk_size=64, A_log=None, dt_bias=None, transpose_state_layout=None):
    """KDA with the signature of FLA's ``fla.ops.kda.chunk_kda``: a drop-in for the Megatron
    (Megatron-Bridge Kimi K3) call site. Returns ``(o, None)`` with o bf16 [B, T, H, V].

    A layout adapter over :func:`kimi_delta_attention`, bit-identical to calling it on the
    packed THD tensors: [B, T, H, D] is flattened to [B*T, H, D] (B > 1 without cu_seqlens
    becomes B packed sequences of T tokens). Gate contracts, as in FLA:

    * ``use_gate_in_kernel=True`` with ``lower_bound`` (``safe_gate``): g is the raw gate,
      activated in kernel as ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` (Kimi K3);
    * ``use_gate_in_kernel=False``: g is the per-token log decay (>= -5).

    ``use_beta_sigmoid_in_kernel`` applies ``sigmoid`` (x2 with ``allow_neg_eigval``) to beta
    on the host. ``disable_recompute`` keeps FLA's meaning (see :func:`kimi_delta_attention`;
    not passing it selects the measured policy). ``chunk_size`` and ``cu_seqlens_cpu`` are FLA
    tuning knobs with no effect here. ``cp_context`` (FLA's chunkwise context parallelism) takes
    each rank's start state from the ranks before it (see :class:`_KdaCpFunction`). Not supported
    (NotImplementedError): intermediate states, FLA's softplus gate (no ``lower_bound``).
    """
    if transpose_state_layout is not None:  # FLA's deprecated name of state_v_first
        if state_v_first:
            raise ValueError("chunk_kda: cannot pass both state_v_first and transpose_state_layout")
        state_v_first = bool(transpose_state_layout)
    if return_intermediate_states:
        raise NotImplementedError("chunk_kda: intermediate states are not supported")
    if cp_context is not None:
        if initial_state is not None or output_final_state:
            raise ValueError("chunk_kda: initial_state / output_final_state are not supported with cp_context")
        if cp_context.cu_seqlens is None:
            raise ValueError("chunk_kda: cp_context requires cu_seqlens")
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu
    if q.dim() != 4:
        raise ValueError(f"chunk_kda: expected [B, T, H, D] inputs, got {tuple(q.shape)}")
    B, T, H, K = q.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError("chunk_kda: cu_seqlens requires B == 1 (packed sequences)")
    if B > 1:
        cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=q.device)
    flat = lambda t: t.reshape(B * T, *t.shape[2:])
    if use_beta_sigmoid_in_kernel:
        beta = (2.0 if allow_neg_eigval else 1.0) * torch.sigmoid(beta.float())
    if use_gate_in_kernel:
        if A_log is None or dt_bias is None:
            raise ValueError("chunk_kda: use_gate_in_kernel requires A_log and dt_bias")
        if lower_bound is None:
            raise NotImplementedError("chunk_kda: the gate needs lower_bound (FLA's safe_gate form)")
        gate = {"A_log": A_log.reshape(H), "dt_bias": dt_bias.reshape(H * K), "lower_bound": float(lower_bound)}
    elif A_log is not None or dt_bias is not None:
        raise ValueError("chunk_kda: A_log / dt_bias require use_gate_in_kernel=True")
    else:
        gate = {}
    x = [flat(q), flat(k), flat(v), flat(g), flat(beta)]
    kw = dict(scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, disable_recompute=disable_recompute,
              **gate)
    if initial_state is not None and state_v_first:
        initial_state = initial_state.transpose(-1, -2)
    if cp_context is not None and cp_context.group is not None:
        o, ht = _cp_forward(*x, cp_context, cu_seqlens=cu_seqlens, **kw), None
    else:
        out = kimi_delta_attention(*x, cu_seqlens=cu_seqlens, initial_state=initial_state,
                                   output_final_state=output_final_state, **kw)
        o, ht = out if output_final_state else (out, None)
    if ht is not None and state_v_first:
        ht = ht.transpose(-1, -2)
    return o.view(B, T, H, v.shape[-1]), ht


class _KdaCpFunction(torch.autograd.Function):
    """One rank's share of a chunkwise context-parallel KDA call (FLA's ``cp_context``).

    The recurrence is affine in the state, so this rank's tokens map a start state S to
    ``M @ S + S_loc``: S_loc is its final state from a zero state, M (K x K) its final state from
    the identity with the value term zeroed. Every rank computes both without communicating
    (the forward's phases 1 and 2, which reuse one preprocessing), all-gathers ``[M | S_loc]``,
    composes the summaries of the ranks before it in fp32 into its start state h0, and runs the
    ordinary recurrence from h0 (phase 3, on the same preprocessing).

    The backward mirrors it: ``dh0 = M^T dht + D``, where D is this rank's dh0 from a zero final-
    state gradient (phase 1), so every rank's dht is the suffix composition of the later ranks'
    (M, D). The gradients of this rank's inputs -- including the dependence of M and S_loc on
    k, beta and g -- all come out of its own backward from that dht (phase 3); the summary passes
    are value-only and are not differentiated, which is why autograd does not run through them.

    Packed sequences (contiguous layout): a rank holds N local sequences, of which only the first
    can continue one from the ranks before it and only the last can run on into the ranks after
    it. So the forward summary is the last local sequence's and h0 seeds the first; the backward
    summary D is the first local sequence's and the merged dht seeds the last. ``pre`` / ``post``
    count the neighbouring ranks that share those two sequences. M is read only from a rank that
    lies strictly inside one sequence, which is then its only local sequence.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, scale, lower_bound, l2norm, grad_mode,
                disable_recompute, group, pre, post):
        import torch.distributed as dist
        needs_bwd = grad_mode and any(ctx.needs_input_grad)
        gate = a_log is not None
        if disable_recompute is None:
            save_h = _save_h(needs_bwd, q.shape[1])
            save_qkn = _save_qkn(needs_bwd, l2norm)
            keep = _keep_state(needs_bwd) and not save_h
        else:
            save_h, save_qkn, keep = needs_bwd and disable_recompute, needs_bwd and l2norm, False
        save_qkn = save_qkn and q.numel() > 0
        H, K, V = q.shape[1], q.shape[2], v.shape[2]
        if K != V:
            raise NotImplementedError("chunk_kda: cp_context needs K == V")
        args = (q, k, v, g, beta, a_log, dt_bias, cu_seqlens)
        base = dict(scale=scale, lower_bound=lower_bound, gate_in_kernel=gate, l2norm_qk=l2norm)
        # Phase 1: the preprocessing (which phases 2 and 3 reuse) plus a state-only recurrence --
        # or phase 4, the preprocessing alone, on a rank whose last sequence ends here: no rank
        # reads its summary.
        world, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
        nseq = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
        wants_summary = post > 0
        # A rank inside one sequence (CP >= 3) also needs the transition matrix M: phase 6 carries
        # [S_loc | M] as one K x (V + K) state through one recurrence pass, M from I with a zero value term.
        with_m = nseq == 1 and pre > 0 and post > 0
        eye = (torch.eye(K, device=q.device, dtype=torch.float32).expand(1, H, K, K).contiguous()
               if with_m else None)
        f1 = _kda_forward(*args, **base, save_qkn=save_qkn, output_final_state=wants_summary,
                          initial_state=eye, cp_phase=6 if with_m else (1 if wants_summary else 4))
        o_buf, q_rstd, k_rstd, aqk, akk, qn, kn = f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6]
        ws = f1[10]
        s_loc = (f1[9][nseq - 1:nseq] if wants_summary  # with_m: nseq == 1, f1[9] is [S_loc; M]
                 else q.new_zeros(1, H, K, V, dtype=torch.float32))
        # Phase 2: the transition matrix, only on the ranks a later rank composes it from (the
        # first summary of a composition multiplies a zero state, the last rank's is never read).
        reuse = dict(cp_workspace=ws, cp_out=o_buf)
        if with_m:
            m = f1[9][1:]
        else:
            m = s_loc.new_zeros(1, H, K, K)
        # Only M of the ranks inside a sequence is ever read: at its first rank M multiplies a zero
        # state, at its last a zero dht. Without such ranks (CP2) the gather carries S_loc alone.
        need_m = world > 2
        hm = torch.cat([m[0], s_loc[0]], dim=-1).contiguous() if need_m else s_loc[0].contiguous()
        gathered = hm.new_empty(world, *hm.shape)
        dist.all_gather_into_tensor(gathered, hm, group=group)
        h = None
        for j in range(rank - pre, rank):
            sj = gathered[j, ..., K:] if need_m else gathered[j]
            h = sj if h is None else torch.matmul(gathered[j, ..., :K], h) + sj
        h0 = hm.new_zeros(nseq, H, K, V)
        if h is not None:
            h0[0] = h
        # Phase 3: the ordinary recurrence from h0 on phase 1's preprocessing.
        f3 = _kda_forward(*args, **base, keep_state=keep, save_h=save_h, initial_state=h0, cp_phase=3, **reuse)
        o, saved_h, state = f3[0], f3[7], f3[8]
        q_sv, k_sv = (None, None) if save_qkn else (q, k)
        ctx.save_for_backward(q_sv, k_sv, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk,
                              qn, kn, saved_h, state, h0, gathered)
        ctx.scale, ctx.lower_bound, ctx.gate, ctx.l2norm = scale, lower_bound, gate, l2norm
        ctx.group, ctx.rank, ctx.world, ctx.need_m = group, rank, world, need_m
        ctx.pre, ctx.post = pre, post
        ctx.set_materialize_grads(False)
        return o

    @staticmethod
    def backward(ctx, d_o):
        import torch.distributed as dist
        (q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, saved_h, state, h0,
         gathered) = ctx.saved_tensors
        if d_o is None:
            d_o = torch.zeros_like(v)
        args = (d_o.contiguous(), q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn,
                saved_h, state)
        base = dict(scale=ctx.scale, lower_bound=ctx.lower_bound, gate_in_kernel=ctx.gate, l2norm_qk=ctx.l2norm,
                    initial_state=h0)
        # Phase 1: the preprocessing plus the state recurrence from a zero dht -> this rank's D --
        # or phase 4, the preprocessing alone, on a rank whose first sequence starts here: no rank reads its D.
        rank, world = ctx.rank, ctx.world
        wants_summary = ctx.pre > 0
        b1 = _kda_backward(*args, **base, dht=None, output_dh0=wants_summary,
                           cp_phase=1 if wants_summary else 4)
        ws, dqk = b1[8], b1[9]
        K = gathered.shape[2]
        shape = (gathered.shape[1], K, gathered.shape[3] - K) if ctx.need_m else tuple(gathered.shape[1:])
        d_loc = b1[7][0] if wants_summary else gathered.new_zeros(shape)
        dgath = gathered.new_empty(world, *shape)
        dist.all_gather_into_tensor(dgath, d_loc.contiguous(), group=ctx.group)
        x = None
        for j in range(rank + ctx.post, rank, -1):   # dht_r = M_{r+1}^T dht_{r+1} + D_{r+1}, zero past the sequence end
            x = dgath[j] if x is None else torch.matmul(gathered[j, ..., :K].transpose(-1, -2), x) + dgath[j]
        dht = None
        if x is not None:
            dht = h0.new_zeros(h0.shape)
            dht[-1] = x
        # Phase 2 (the backward's second phase): the recurrence again from the merged dht, then
        # the input gradients, on phase 1's preprocessing and in its buffers.
        dq, dk, dv, dg, dbeta, da_log, ddt_bias = _kda_backward(
            *args, **base, dht=dht, output_dh0=False, cp_phase=3, cp_workspace=ws,
            cp_outs=[dqk, b1[2], b1[3], b1[4], b1[5], b1[6]])[:7]
        if not ctx.gate:
            da_log = ddt_bias = None
        return (dq, dk, dv, dg, dbeta, da_log, ddt_bias, None, None, None, None, None, None, None, None, None)


def _cp_forward(q, k, v, g, beta, cp, *, cu_seqlens=None, scale=None, A_log=None, dt_bias=None, lower_bound=-5.0,
                use_qk_l2norm_in_kernel=True, check_decay=True, disable_recompute=None):
    """Chunkwise context parallelism (FLA's ``cp_context``): this rank's output o [T, H, V], its
    start state composed from the ranks before it (see :class:`_KdaCpFunction`). Same input
    contract as :func:`kimi_delta_attention`; ``cp`` is FLA's CP context."""
    q, k, v, g = (_thd(t, n) for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g")))
    beta_in = _thd(beta.unsqueeze(-1), "beta").squeeze(-1) if beta.dim() == 3 else beta.contiguous()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if getattr(cp, "layout", "contiguous") != "contiguous":
        raise NotImplementedError("chunk_kda: cp_context supports the contiguous layout only")
    if cu_seqlens is not None:
        # One local sequence is the dense layout: bit-identical and on the faster kernel instances.
        cu_seqlens = None if cu_seqlens.numel() == 2 else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    if A_log is not None:
        if dt_bias is None:
            raise ValueError("chunk_kda: A_log requires dt_bias")
        g, a32, d32 = g.to(torch.bfloat16), A_log.float().contiguous(), dt_bias.float().contiguous()
    else:
        g, a32, d32 = g.float(), None, None
        if check_decay and g.numel() and float(g.detach().amin()) < DECAY_LOWER_BOUND:
            raise ValueError("chunk_kda: per-token log decay below -5 is outside the supported range")
        lower_bound = 0.0
    return _KdaCpFunction.apply(q, k, v, g, beta_in.float(), a32, d32, cu_seqlens, float(scale), float(lower_bound),
                                bool(use_qk_l2norm_in_kernel), torch.is_grad_enabled(), disable_recompute,
                                cp.group, 0 if cp.is_first_rank else int(cp.pre_num_ranks),
                                0 if cp.is_last_rank else int(cp.post_num_ranks))


# ----------------------------------------------------------------------------- reference
def kda_reference(q, k, v, g, beta, *, cu_seqlens=None, scale=None, A_log=None, dt_bias=None, lower_bound=-5.0,
                  use_qk_l2norm_in_kernel=True, dtype=torch.float64, initial_state=None, output_final_state=False):
    """PyTorch reference of :func:`kimi_delta_attention` (same arguments): the token-by-token
    recurrence in ``dtype`` (fp64 by default), differentiable. Returns o [T, H, V] in dtype, or
    (o, final_state [N, H, K, V]) with ``output_final_state``."""
    q, k, v, g = (_thd(t, n).to(dtype) for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g")))
    beta = beta.squeeze(0).to(dtype) if beta.dim() == 3 else beta.to(dtype)
    T, H, K = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if A_log is not None:
        g = lower_bound * torch.sigmoid(A_log.to(dtype).exp().view(1, H, 1) * (g + dt_bias.to(dtype).view(1, H, K)))
    alpha = g.exp()
    bounds = [0, T] if cu_seqlens is None else [int(x) for x in cu_seqlens.tolist()]
    outs, finals = [], []
    for n, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = q.new_zeros(H, K, v.shape[-1]) if initial_state is None else initial_state[n].to(dtype)
        for t in range(s, e):
            state = alpha[t].unsqueeze(-1) * state
            residual = v[t] - torch.einsum("hk,hkv->hv", k[t], state)
            state = state + beta[t].view(H, 1, 1) * k[t].unsqueeze(-1) * residual.unsqueeze(-2)
            outs.append(torch.einsum("hk,hkv->hv", q[t] * scale, state))
        finals.append(state)
    o = torch.stack(outs, 0) if outs else v.new_zeros(0, H, v.shape[-1])
    return (o, torch.stack(finals, 0)) if output_final_state else o


def kda_chunked_reference(q, k, v, g, beta, *, cu_seqlens=None, scale=None, A_log=None, dt_bias=None,
                          lower_bound=-5.0, use_qk_l2norm_in_kernel=True, chunk=64):
    """PyTorch composite implementation (fp32, chunked WY form, forward only): the
    benchmark baseline. Mathematically equal to :func:`kda_reference`."""
    q, k, v, g = (_thd(t, n).float() for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g")))
    beta = (beta.squeeze(0) if beta.dim() == 3 else beta).float()
    T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if A_log is not None:
        g = lower_bound * torch.sigmoid(A_log.float().exp().view(1, H, 1) * (g + dt_bias.float().view(1, H, K)))
    bounds = [0, T] if cu_seqlens is None else [int(x) for x in cu_seqlens.tolist()]
    o = torch.empty(T, H, V, device=q.device, dtype=torch.float32)
    eye = torch.eye(chunk, device=q.device)
    tri_strict = torch.ones(chunk, chunk, device=q.device).tril(-1)
    tri = torch.ones(chunk, chunk, device=q.device).tril()
    for s, e in zip(bounds[:-1], bounds[1:]):
        state = q.new_zeros(H, K, V)
        for c0 in range(s, e, chunk):
            c1 = min(c0 + chunk, e)
            n = c1 - c0
            qc, kc, vc = (x[c0:c1].transpose(0, 1) for x in (q, k, v))             # [H, n, *]
            gc = g[c0:c1].transpose(0, 1).cumsum(1)                                  # [H, n, K]
            bc = beta[c0:c1].transpose(0, 1)                                          # [H, n]
            # pairwise decays exp(G_i - G_j) for j <= i (masked before exp: no overflow)
            diff = gc.unsqueeze(2) - gc.unsqueeze(1)                                  # [H, n, n, K]
            diff = diff.masked_fill(tri[:n, :n].view(1, n, n, 1) == 0, float("-inf"))
            dec = diff.exp()
            kk = torch.einsum("hik,hjk,hijk->hij", kc, kc, dec)
            akk = bc.unsqueeze(-1) * kk * tri_strict[:n, :n]
            tinv = torch.linalg.solve_triangular(eye[:n, :n] + akk, eye[:n, :n].expand(H, n, n), upper=False)
            u = tinv @ (vc * bc.unsqueeze(-1))
            w = tinv @ (kc * gc.exp() * bc.unsqueeze(-1))
            vnew = u - w @ state
            qg = qc * gc.exp() * scale
            aqk = torch.einsum("hik,hjk,hijk->hij", qc * scale, kc, dec)
            o[c0:c1] = (qg @ state + aqk @ vnew).transpose(0, 1)
            glast = gc[:, -1:, :]
            state = glast.transpose(1, 2).exp() * state + (kc * (glast - gc).exp()).transpose(1, 2) @ vnew
    return o
