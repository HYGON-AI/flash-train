# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""mHC compatibility layer: the ``fused_mhc`` / Megatron calling convention.

``flash_train.torch`` exposes the mHC operators as pure operators. The standalone
``fused_mhc`` package and Megatron's ``fused_mhc_kernels`` module instead call
``torch.autograd.Function`` subclasses whose ``forward``/``backward`` pair the
training loop drives directly.

This module is that adapter: the same class names, the same forward arguments,
and the same backward return order, each implemented on top of the
``flash_train.torch`` operators. Porting a call site is then an import change::

    from flash_train.torch.mhc_wrappers import (
        FusedSinkhorn, FusedHAggregate, FusedHPostBDA, FusedProjRmsComputeH,
    )

Nothing here is a kernel: every method runs the library's staged path, so the
autograd wrappers and the pure operators share one implementation, one
selection and one cache. The autograd bookkeeping mirrors the standalone
package's, including that ``h_post_bda`` stores the bias in the context and
that the sinkhorn forward hands its FP32 state to the backward.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._flash_train_torch import (
    h_aggregate_bwd as _h_aggregate_bwd,
    h_aggregate_fwd as _h_aggregate_fwd,
    h_post_bda_bwd as _h_post_bda_bwd,
    h_post_bda_fwd as _h_post_bda_fwd,
    proj_rms_bwd as _proj_rms_bwd,
    proj_rms_compute_h_bwd as _proj_rms_compute_h_bwd,
    proj_rms_compute_h_fwd as _proj_rms_compute_h_fwd,
    proj_rms_fwd as _proj_rms_fwd,
    sinkhorn_bwd as _sinkhorn_bwd,
    sinkhorn_fwd as _sinkhorn_fwd,
)


class FusedSinkhorn(torch.autograd.Function):
    """Sinkhorn-Knopp projection with the library's fused backward."""

    @staticmethod
    def forward(
        ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> Tensor:
        output, m_init = _sinkhorn_fwd(
            logits=input_logits, num_iterations=int(num_iterations), eps=float(eps)
        )
        ctx.save_for_backward(m_init)
        ctx.num_iterations = int(num_iterations)
        ctx.eps = float(eps)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (m_init,) = ctx.saved_tensors
        grad_input = _sinkhorn_bwd(
            grad_output=grad_output.contiguous(),
            m_init=m_init,
            num_iterations=ctx.num_iterations,
            eps=ctx.eps,
        )
        return grad_input, None, None


class FusedHAggregate(torch.autograd.Function):
    """Weighted n-stream to 1-stream aggregation."""

    @staticmethod
    def forward(ctx, x: Tensor, h_pre: Tensor) -> Tensor:
        h_float = h_pre.float().contiguous()
        output = _h_aggregate_fwd(x=x.contiguous(), h_pre=h_float)
        ctx.save_for_backward(x, h_float)
        ctx.h_dtype = h_pre.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, h_pre = ctx.saved_tensors
        grad_x, grad_h = _h_aggregate_bwd(
            grad_aggregated=grad_output.contiguous(), x=x.contiguous(), h_pre=h_pre
        )
        return grad_x, grad_h.to(ctx.h_dtype)


class FusedHPostBDA(torch.autograd.Function):
    """``h_resᵀ @ residual + h_post * (x + bias)``."""

    @staticmethod
    def forward(
        ctx,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        output = _h_post_bda_fwd(
            h_res=h_res,
            residual=original_residual,
            h_post=h_post,
            x=x,
            bias=None if bias is None else bias.float().contiguous(),
        )
        ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.has_bias:
            h_res, original_residual, h_post, x, bias = ctx.saved_tensors
        else:
            h_res, original_residual, h_post, x, _ = ctx.saved_tensors
            bias = None
        grad_residual, grad_x, grad_h_res, grad_h_post, grad_bias = _h_post_bda_bwd(
            grad_output=grad_output.contiguous(),
            residual=original_residual,
            x=x,
            h_res=h_res,
            h_post=h_post,
            bias=None if bias is None else bias.float().contiguous(),
            want_bias_gradient=ctx.has_bias,
        )
        if bias is not None:
            grad_bias = grad_bias.to(dtype=bias.dtype)
        else:
            grad_bias = None
        # The forward's argument order is (h_res, residual, h_post, x, bias).
        return grad_h_res, grad_residual, grad_h_post, grad_x, grad_bias


class FusedProjRms(torch.autograd.Function):
    """Projection plus ``r = 1 / (|x| / sqrt(K) + eps)``."""

    @staticmethod
    def forward(
        ctx, x: Tensor, weight: Tensor, eps: float = 1e-6
    ) -> Tuple[Tensor, Tensor]:
        proj, r = _proj_rms_fwd(x=x, weight=weight, eps=float(eps))
        ctx.save_for_backward(x, weight, r)
        ctx.eps = float(eps)
        return proj, r

    @staticmethod
    def backward(ctx, grad_proj: Tensor, grad_r: Tensor):
        x, weight, r = ctx.saved_tensors
        grad_x, grad_weight = _proj_rms_bwd(
            grad_proj=grad_proj.contiguous(),
            grad_r=grad_r.contiguous(),
            x=x,
            weight=weight,
            r=r,
            eps=ctx.eps,
        )
        return grad_x, grad_weight, None


class FusedProjRmsComputeH(torch.autograd.Function):
    """Projection, RMS factor, and the mHC activation split (``compute_h``)."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        bias: Tensor,
        n: int,
        eps: float = 1e-6,
        compute_h_eps: float = 1e-6,
    ):
        h_pre, h_post, h_res, r, proj = _proj_rms_compute_h_fwd(
            x=x,
            weight=weight,
            alpha_pre=alpha_pre.float().contiguous(),
            alpha_post=alpha_post.float().contiguous(),
            alpha_res=alpha_res.float().contiguous(),
            bias=bias.float().contiguous(),
            n=int(n),
            eps=float(eps),
            compute_h_eps=float(compute_h_eps),
        )
        ctx.save_for_backward(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post, proj, r
        )
        ctx.n = int(n)
        ctx.eps = float(eps)
        ctx.compute_h_eps = float(compute_h_eps)
        return h_pre, h_post, h_res, r

    @staticmethod
    def backward(
        ctx, grad_h_pre: Tensor, grad_h_post: Tensor, grad_h_res: Tensor, grad_r: Tensor
    ):
        x, weight, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post, proj, r = (
            ctx.saved_tensors
        )
        grad_x, grad_weight, grad_alpha, grad_bias = _proj_rms_compute_h_bwd(
            grad_h_pre=grad_h_pre.contiguous(),
            grad_h_post=grad_h_post.contiguous(),
            grad_h_res=grad_h_res.contiguous(),
            grad_r=grad_r.contiguous(),
            h_pre=h_pre,
            h_post=h_post,
            proj=proj,
            r=r,
            x=x,
            weight=weight,
            alpha_pre=alpha_pre.float().contiguous(),
            alpha_post=alpha_post.float().contiguous(),
            alpha_res=alpha_res.float().contiguous(),
            n=ctx.n,
            eps=ctx.eps,
            compute_h_eps=ctx.compute_h_eps,
        )
        grad_bias = grad_bias.to(bias.dtype)
        return (
            grad_x,
            grad_weight,
            grad_alpha[0].reshape(alpha_pre.shape).to(alpha_pre.dtype),
            grad_alpha[1].reshape(alpha_post.shape).to(alpha_post.dtype),
            grad_alpha[2].reshape(alpha_res.shape).to(alpha_res.dtype),
            grad_bias,
            None,
            None,
            None,
        )


def fused_sinkhorn(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6
) -> Tensor:
    """Project logits to a doubly stochastic matrix via Sinkhorn-Knopp."""
    return FusedSinkhorn.apply(input_logits, num_iterations, eps)


def fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Weighted n-stream to 1-stream aggregation."""
    return FusedHAggregate.apply(x, h_pre)


def fused_h_post_bda(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    """Fused ``h_resᵀ @ residual + h_post * (x + bias)``."""
    return FusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)


def fused_proj_rms(
    x: Tensor, weight: Tensor, eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """Fused projection plus RMS normalization."""
    return FusedProjRms.apply(x, weight, eps)


def fused_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
    compute_h_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Projection, RMS norm, and the mHC ``compute_h`` split."""
    return FusedProjRmsComputeH.apply(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, compute_h_eps
    )


def is_hip_available() -> bool:
    """Return True: the mHC operators are served by the ported HCU kernels."""
    return True


__all__ = [
    "FusedHAggregate",
    "FusedHPostBDA",
    "FusedProjRms",
    "FusedProjRmsComputeH",
    "FusedSinkhorn",
    "fused_h_aggregate",
    "fused_h_post_bda",
    "fused_proj_rms",
    "fused_proj_rms_compute_h",
    "fused_sinkhorn",
    "is_hip_available",
]
