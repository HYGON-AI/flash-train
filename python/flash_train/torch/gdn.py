# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable GDN operators for contiguous BHLD tensors."""

import torch

from ._flash_train_torch import gdn2_backward, gdn2_forward, gdn_backward, gdn_forward


def _unsupported(**options):
    for name, value in options.items():
        if value is not None and value is not False:
            raise ValueError(f"{name} is unsupported by FlashTrain")


def _backward_shape(q, v):
    return (
        q.dtype == torch.bfloat16
        and q.ndim == v.ndim == 4
        and q.shape[1] >= 512
        and q.shape[1] % 64 == 0
        and q.shape[-1] in (64, 128)
        and v.shape[-1] in (64, 128)
    )


class _Gdn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state):
        ctx.scale = scale
        ctx.has_state = initial_state is not None
        ctx.save_for_backward(q, k, v, g, beta, *(() if initial_state is None else (initial_state,)))
        ctx.set_materialize_grads(False)
        return gdn_forward(q, k, v, g, beta.float().contiguous(), scale, initial_state, output_final_state)

    @staticmethod
    def backward(ctx, dout, dht):
        saved = ctx.saved_tensors
        q, k, v, g, beta = saved[:5]
        h0 = saved[5] if ctx.has_state else None
        dq, dk, dv, dg, dbeta, dh0 = gdn_backward(
            q, k, v, g, beta.float().contiguous(),
            torch.zeros_like(v) if dout is None else dout.contiguous(),
            ctx.scale, h0, None if dht is None else dht.contiguous(), ctx.has_state,
        )
        return dq, dk, dv, dg, dbeta.to(beta.dtype), None, dh0, None


class _Gdn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, scale):
        ctx.scale = scale
        ctx.save_for_backward(q, k, v, g, b, w)
        ctx.set_materialize_grads(False)
        return gdn2_forward(q, k, v, g.float().contiguous(), b.float().contiguous(),
                            w.float().contiguous(), scale)

    @staticmethod
    def backward(ctx, dout, _final_state):
        q, k, v, g, b, w = ctx.saved_tensors
        dq, dk, dv, dg, db, dw = gdn2_backward(
            q, k, v, g.float().contiguous(), b.float().contiguous(), w.float().contiguous(),
            torch.zeros_like(v) if dout is None else dout.contiguous(), ctx.scale,
        )
        return dq, dk, dv, dg.to(g.dtype), db.to(b.dtype), dw.to(w.dtype), None


@torch.compiler.disable
def chunk_gated_delta_rule(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
    cu_seqlens_cpu=None, cp_context=None, **kwargs,
):
    _unsupported(
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, cp_context=cp_context,
        use_gate_in_kernel=kwargs.pop("use_gate_in_kernel", False),
        A_log=kwargs.pop("A_log", None), dt_bias=kwargs.pop("dt_bias", None),
    )
    if kwargs.pop("chunk_size", 64) != 64:
        raise ValueError("FlashTrain only supports chunk_size=64")
    if kwargs:
        raise TypeError(f"unsupported GDN arguments: {', '.join(kwargs)}")
    if any(t.requires_grad for t in (q, k, v, g, beta) if isinstance(t, torch.Tensor)) or (
        initial_state is not None and initial_state.requires_grad
    ):
        if not _backward_shape(q, v):
            raise ValueError("FlashTrain GDN backward requires bf16, S>=512, S%64=0, K/V in {64,128}")
    return _Gdn.apply(q, k, v, g, beta, scale, initial_state, output_final_state)


@torch.compiler.disable
def chunk_gdn2(
    q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, cu_seqlens=None,
    cu_seqlens_cpu=None, safe_gate=False, lower_bound=None, disable_recompute=False,
    return_intermediate_states=False, state_v_first=False, **kwargs,
):
    _unsupported(
        initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu, safe_gate=safe_gate,
        lower_bound=lower_bound, disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states, state_v_first=state_v_first,
    )
    if kwargs.pop("chunk_size", 64) != 64:
        raise ValueError("FlashTrain only supports chunk_size=64")
    if kwargs:
        raise TypeError(f"unsupported GDN2 arguments: {', '.join(kwargs)}")
    if any(t.requires_grad for t in (q, k, v, g, b, w) if isinstance(t, torch.Tensor)):
        if not _backward_shape(q, v):
            raise ValueError("FlashTrain GDN2 backward requires bf16, S>=512, S%64=0, K/V in {64,128}")
    return _Gdn2.apply(q, k, v, g, b, w, scale)
