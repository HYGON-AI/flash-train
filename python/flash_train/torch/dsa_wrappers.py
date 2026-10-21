# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSA compatibility layer: the ``hip_dsa`` / ``cudnn.DSA`` calling convention.

``flash_train.torch`` itself exposes pure operators: every call takes input tensors
and returns outputs allocated by the binding. Frameworks that were written against ``cudnn.DSA`` or
``hip_dsa`` instead call *named wrappers* whose arguments and return values
follow the vendor convention (keyword-only scales, optional metadata, and
results returned as a dict or a tuple).

This module is that adapter: it keeps the vendor-facing shapes and names and
turns each one into the corresponding ``flash_train.torch`` operator. Porting a
call site is then an import change::

    from flash_train.torch import DSA                      # instead of hip_dsa
    from flash_train.torch import sparse_attn_forward      # instead of hip_dsa

The adapters only normalize arguments and package results. A shape the library does not implement raises, exactly as the
HIP-only ``hip_dsa`` package raised when a kernel was missing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._flash_train_torch import (
    compactify as _compactify_op,
    csa_compressor_bwd as _csa_compressor_bwd_op,
    csa_compressor_fwd as _csa_compressor_fwd_op,
    dense_attn_score as _dense_attn_score_op,
    dense_attn_score_thd as _dense_attn_score_thd_op,
    sparse_attn_fwd as _sparse_attn_fwd_op,
    sparse_attn_fwd_indexer as _sparse_attn_fwd_indexer_op,
    sparse_attn_bwd as _sparse_attn_bwd_op,
    sparse_attn_target as _sparse_attn_target_op,
    sparse_indexer_predict as _sparse_indexer_predict_op,
    indexer_bwd_dense as _indexer_bwd_dense_op,
    indexer_bwd_dense_thd as _indexer_bwd_dense_thd_op,
    indexer_bwd_sparse as _indexer_bwd_sparse_op,
    indexer_fwd as _indexer_fwd_op,
    indexer_fwd_thd as _indexer_fwd_thd_op,
    indexer_topk as _indexer_topk_op,
    indexer_topk_thd as _indexer_topk_thd_op,
    dense_indexer_score as _dense_indexer_score_op,
    dense_indexer_score_thd as _dense_indexer_score_thd_op,
)


def _contiguous_inputs(*tensors):
    return tuple(t.contiguous() for t in tensors)


def _check_thd_metadata(cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv) -> None:
    if cu_seqlens_q is None or cu_seqlens_kv is None:
        raise ValueError("THD mode requires both cu_seqlens_q and cu_seqlens_kv")
    if max_seqlen_kv is None:
        raise ValueError("THD mode requires max_seqlen_kv")


# ---------------------------------------------------------------------------
# Sparse attention forward (FlashMLA replacement)
# ---------------------------------------------------------------------------


def sparse_attn_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    softmax_scale: float = 1.0,
    topk_length: Optional[torch.Tensor] = None,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out (T,H,512), lse (T,H) f32)``; only ``d_v=512`` is supported."""
    q, kv = _contiguous_inputs(q, kv)
    if d_v != 512:
        raise ValueError("sparse attention forward only supports d_v=512")
    out, lse = _sparse_attn_fwd_op(
        q=q,
        kv=kv,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    return out, lse


def sparse_attn_fwd_indexer(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    softmax_scale: float = 1.0,
    topk_length: Optional[torch.Tensor] = None,
    d_v: int = 512,
    indexer_topk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused forward with the indexer region.

    Returns ``(out (T,H,d_v), max_logits (T,H), lse (T,H), lse_indexer (T,H))``
    with ``max_logits``/``lse``/``lse_indexer`` in FP32, matching the
    ``hip_dsa`` wrapper.
    """
    q, kv = _contiguous_inputs(q, kv)
    if d_v != 512:
        raise ValueError("sparse attention forward only supports d_v=512")
    if indexer_topk <= 0:
        raise ValueError("sparse_attn_fwd_indexer requires indexer_topk > 0")
    out, max_logits, lse, lse_indexer = _sparse_attn_fwd_indexer_op(
        q=q,
        kv=kv,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        softmax_scale=softmax_scale,
        indexer_topk=indexer_topk,
        topk_length=topk_length,
    )
    return out, max_logits, lse, lse_indexer


# ---------------------------------------------------------------------------
# The cudnn.DSA-shaped namespace
# ---------------------------------------------------------------------------


class DSANamespace:
    """``cudnn.DSA``-compatible attribute access backed by ``flash_train.torch``."""

    _is_hip_backend = True
    _backend_name = "flash_train.torch"

    # -- sparse attention ---------------------------------------------------

    @staticmethod
    def sparse_attention_backward_wrapper(
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        softmax_scale: float = 1.0,
        topk_length: Optional[torch.Tensor] = None,
    ) -> dict:
        """Sparse attention backward; returns ``{'dq', 'dkv', 'd_sink'}``."""
        q, kv, out, dout = _contiguous_inputs(q, kv, out, dout)
        dq, dkv, dsink = _sparse_attn_bwd_op(
            q=q,
            kv=kv,
            out=out,
            dout=dout,
            lse=lse,
            attn_sink=attn_sink,
            topk_idxs=topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
        )
        return {"dq": dq, "dkv": dkv, "d_sink": dsink}

    # -- indexer ------------------------------------------------------------

    @staticmethod
    def indexer_forward_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Indexer forward: dense scores over all KV.

        BSHD by default; THD when ``cu_seqlens_q``/``cu_seqlens_kv`` are
        supplied. ``cu_seqlens_k``/``max_seqlen_k`` are accepted as aliases,
        matching the official wrapper convention.
        """
        q, k, w = _contiguous_inputs(q, k, w)
        del qhead_per_kv_head  # the kernels are MQA; the argument is API compatibility
        if cu_seqlens_k is not None and cu_seqlens_kv is None:
            cu_seqlens_kv = cu_seqlens_k
        if max_seqlen_k is not None and max_seqlen_kv is None:
            max_seqlen_kv = max_seqlen_k
        is_thd = cu_seqlens_q is not None or cu_seqlens_kv is not None
        if not is_thd and q_causal_offsets is not None:
            raise ValueError("q_causal_offsets requires THD metadata")
        if is_thd:
            _check_thd_metadata(cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv)
            scores = _indexer_fwd_thd_op(
                q=q,
                k=k,
                w=w,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                ratio=ratio,
                sm_scale=sm_scale,
                max_seqlen_kv=int(max_seqlen_kv),
                q_causal_offsets=q_causal_offsets,
            )
        else:
            scores = _indexer_fwd_op(q=q, k=k, w=w, ratio=ratio, sm_scale=sm_scale)
        return {"scores": scores}

    @staticmethod
    def dense_indexer_score(
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        *,
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Dense indexer score; returns ``{'out', 'denom'}``, both FP32."""
        q, k, w = _contiguous_inputs(q, k, w)
        del qhead_per_kv_head, max_seqlen_q
        is_thd = cu_seqlens_q is not None or cu_seqlens_kv is not None
        if not is_thd and q_causal_offsets is not None:
            raise ValueError("q_causal_offsets requires THD metadata")
        if is_thd:
            _check_thd_metadata(cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv)
            out, denom = _dense_indexer_score_thd_op(
                q=q,
                k=k,
                w=w,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                ratio=ratio,
                sm_scale=sm_scale,
                max_seqlen_kv=int(max_seqlen_kv),
                q_causal_offsets=q_causal_offsets,
            )
        else:
            out, denom = _dense_indexer_score_op(
                q=q, k=k, w=w, ratio=ratio, sm_scale=sm_scale
            )
        return {"out": out, "denom": denom}

    @staticmethod
    def sparse_indexer_predict(
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        softmax_scale: float = 1.0,
        topk_indices_global: bool = False,
    ) -> dict:
        """Sparse indexer predict; returns ``{'predict'}``."""
        q, k, w = _contiguous_inputs(q, k, w)
        predict = _sparse_indexer_predict_op(
            q=q,
            k=k,
            w=w,
            topk_idxs=topk_idxs,
            sm_scale=softmax_scale,
            topk_indices_global=topk_indices_global,
        )
        return {"predict": predict}

    @staticmethod
    def sparse_indexer_score_recompute_wrapper(
        q,
        k,
        w,
        topk_idxs,
        sm_scale=1.0,
        *,
        softmax_scale=None,
        qhead_per_kv_head=None,
        topk_indices_global=False,
    ) -> dict:
        """Recompute sparse indexer scores using the framework scale spelling."""
        return DSANamespace.sparse_indexer_predict(
            q,
            k,
            w,
            topk_idxs,
            softmax_scale=sm_scale if softmax_scale is None else softmax_scale,
            topk_indices_global=topk_indices_global,
        )

    @staticmethod
    def sparse_attn_target(
        q: torch.Tensor,
        k: torch.Tensor,
        lse: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        softmax_scale: float = 1.0,
        topk_indices_global: bool = False,
    ) -> dict:
        """Sparse attention target; returns ``{'target'}``."""
        q, k = _contiguous_inputs(q, k)
        target = _sparse_attn_target_op(
            q=q,
            k=k,
            lse=lse,
            topk_idxs=topk_idxs,
            softmax_scale=softmax_scale,
            topk_indices_global=topk_indices_global,
        )
        return {"target": target}

    @staticmethod
    def sparse_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        topk_idxs,
        softmax_scale=1.0,
        *,
        sm_scale=None,
        qhead_per_kv_head=None,
        topk_indices_global=False,
        topk_length=None,
    ) -> dict:
        """Recompute attention targets over each row's consumed prefix."""
        if topk_length is not None:
            if topk_length.shape != topk_idxs.shape[:-1]:
                raise ValueError("topk_length must match the top-k row dimensions")
            positions = torch.arange(topk_idxs.shape[-1], device=topk_idxs.device)
            topk_idxs = topk_idxs.masked_fill(
                positions >= topk_length.to(device=topk_idxs.device).unsqueeze(-1), -1
            )
        return DSANamespace.sparse_attn_target(
            q,
            k,
            lse,
            topk_idxs,
            softmax_scale=softmax_scale if sm_scale is None else sm_scale,
            topk_indices_global=topk_indices_global,
        )

    @staticmethod
    def dense_attn_score(
        q: torch.Tensor,
        k: torch.Tensor,
        lse: torch.Tensor,
        *,
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Dense attention score; returns ``{'out', 'denom'}``, both FP32."""
        q, k = _contiguous_inputs(q, k)
        del qhead_per_kv_head, max_seqlen_q
        is_thd = cu_seqlens_q is not None or cu_seqlens_kv is not None
        if not is_thd and q_causal_offsets is not None:
            raise ValueError("q_causal_offsets requires THD metadata")
        if is_thd:
            _check_thd_metadata(cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv)
            out, denom = _dense_attn_score_thd_op(
                q=q,
                k=k,
                lse=lse,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                ratio=ratio,
                softmax_scale=sm_scale,
                max_seqlen_kv=int(max_seqlen_kv),
                q_causal_offsets=q_causal_offsets,
            )
        else:
            out, denom = _dense_attn_score_op(
                q=q, k=k, lse=lse, ratio=ratio, softmax_scale=sm_scale
            )
        return {"out": out, "denom": denom}

    @staticmethod
    def dense_indexer_score_recompute_wrapper(
        q,
        k,
        w,
        *,
        ratio=1,
        sm_scale=1.0,
        qhead_per_kv_head=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
    ) -> dict:
        """Framework dense indexer recomputation (ratio defaults to one)."""
        return DSANamespace.dense_indexer_score(
            q,
            k,
            w,
            ratio=ratio,
            sm_scale=sm_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv if max_seqlen_kv is not None else max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
        )

    @staticmethod
    def dense_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        softmax_scale=1.0,
        *,
        ratio=1,
        qhead_per_kv_head=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
    ) -> dict:
        """Framework attention recomputation with positional softmax scale."""
        return DSANamespace.dense_attn_score(
            q,
            k,
            lse,
            ratio=ratio,
            sm_scale=softmax_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv if max_seqlen_kv is not None else max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
        )

    # -- top-K / compactify -------------------------------------------------

    @staticmethod
    def indexer_top_k_wrapper(
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
    ) -> dict:
        """Per-row top-K selection; returns ``{'indices', 'values'}``.

        ``values`` is ``None`` when ``return_val`` is False, matching the
        official wrapper convention.
        """
        indices, values = _indexer_topk_op(
            input_values=input_values,
            seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            order_values=bool(return_val),
        )
        return {"indices": indices, "values": values if return_val else None}

    @staticmethod
    def indexer_top_k_thd_wrapper(
        input_values: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        ratio: int,
        top_k: int,
        *,
        return_val: bool = True,
        q_causal_offsets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Packed-batch top-K; returns ``{'indices', 'values'}``."""
        indices, values = _indexer_topk_thd_op(
            input_values=input_values,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            ratio=ratio,
            top_k=top_k,
            order_values=bool(return_val),
            q_causal_offsets=q_causal_offsets,
        )
        return {"indices": indices, "values": values if return_val else None}

    @staticmethod
    def compactify_wrapper(indices: torch.Tensor) -> dict:
        """Pack the valid selections; returns ``{'indices', 'topk_length'}``."""
        output, topk_length = _compactify_op(indices=indices)
        return {"indices": output, "topk_length": topk_length}

    # -- indexer backward ---------------------------------------------------

    @staticmethod
    def indexer_backward_sparse(
        index_q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        attn_score: torch.Tensor,
        index_score: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        grad_loss: float | torch.Tensor = 1.0,
        topk_indices_global: bool = False,
    ) -> dict:
        """Sparse indexer backward; returns ``{'d_index_q', 'd_weights', 'd_index_k'}``.

        BF16 accepts an FP32 GPU ``grad_loss`` scalar without a host readback.

        ``attn_score`` is updated in place (the kernel reads the target and
        writes the selection gradient through it).
        """
        index_q, weights, index_k = _contiguous_inputs(index_q, weights, index_k)
        _, dq, dw, dk = _indexer_bwd_sparse_op(
            index_q=index_q,
            weights=weights,
            index_k=index_k,
            attn_score=attn_score,
            index_score=index_score,
            topk_idxs=topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            topk_indices_global=topk_indices_global,
        )
        return {"d_index_q": dq, "d_weights": dw, "d_index_k": dk}

    @staticmethod
    def indexer_backward_dense(
        index_q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        attn_score: torch.Tensor,
        attn_l1norm: torch.Tensor,
        index_score: torch.Tensor,
        index_lse: torch.Tensor,
        *,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        grad_loss: float = 1.0,
        ratio: int = 1,
    ) -> dict:
        """Dense indexer backward; returns the three indexer gradients.

        Like ``dense_indexer_backward_wrapper``, the default is uncompressed
        causal attention (ratio=1); pass the compression ratio explicitly.
        """
        index_q, weights, index_k = _contiguous_inputs(index_q, weights, index_k)
        _, dq, dw, dk = _indexer_bwd_dense_op(
            index_q=index_q,
            weights=weights,
            index_k=index_k,
            attn_score=attn_score,
            attn_l1norm=attn_l1norm,
            index_score=index_score,
            index_lse=index_lse,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            ratio=ratio,
        )
        return {"d_index_q": dq, "d_weights": dw, "d_index_k": dk}

    @staticmethod
    def dense_indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        *,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=1.0,
        ratio=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        max_seqlen_kv=None,
        q_causal_offsets=None,
        block_I=None,
    ) -> dict:
        """Dense KL backward, including packed sequences and causal offsets.

        ``block_I`` is a cuDNN tuning hint; flash-train selects its own kernel.
        ``index_score`` is overwritten by the selection gradient.
        """
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_kv
        inputs = dict(
            index_q=index_q,
            weights=weights,
            index_k=index_k,
            attn_score=attn_score,
            attn_l1norm=attn_l1norm,
            index_score=index_score,
            index_lse=index_lse,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            ratio=ratio,
        )
        if cu_seqlens_q is None and cu_seqlens_k is None:
            return DSANamespace.indexer_backward_dense(**inputs)
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "dense indexer backward THD mode requires both cu_seqlens_q and cu_seqlens_k"
            )
        iq, iw, ik = _contiguous_inputs(index_q, weights, index_k)
        inputs.update(index_q=iq, weights=iw, index_k=ik)
        _, dq, dw, dk = _indexer_bwd_dense_thd_op(
            **inputs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            q_causal_offsets=q_causal_offsets,
        )
        return {"d_index_q": dq, "d_weights": dw, "d_index_k": dk}

    @staticmethod
    def indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        *,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=1.0,
        topk_indices_global=False,
        block_I=None,
    ) -> dict:
        """Sparse KL backward; ``block_I`` is a cuDNN-only tuning hint."""
        return DSANamespace.indexer_backward_sparse(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            topk_indices_global=topk_indices_global,
        )

    @staticmethod
    def csa_compressor_forward_wrapper(
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        ratio: int,
        head_dim: int,
        coff: int,
        total_comp: int,
    ) -> dict:
        """Fused gated-pooling forward; returns ``{'out'}``."""
        out = _csa_compressor_fwd_op(
            kv=kv,
            score=score,
            ape=ape,
            cu_seqlens=cu_seqlens,
            cu_seqlens_comp=cu_seqlens_comp,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            total_comp=total_comp,
        )
        return {"out": out}

    @staticmethod
    def csa_compressor_backward_wrapper(
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        grad_out: torch.Tensor,
        ratio: int,
        head_dim: int,
        coff: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused gated-pooling backward; returns ``(gkv, gs, gape)``."""
        gkv, gs, gape = _csa_compressor_bwd_op(
            kv=kv,
            score=score,
            ape=ape,
            cu_seqlens=cu_seqlens,
            cu_seqlens_comp=cu_seqlens_comp,
            grad_compressed=grad_out,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        return gkv, gs, gape


DSA = DSANamespace()

__all__ = ["DSA", "DSANamespace", "sparse_attn_forward", "sparse_attn_fwd_indexer"]
