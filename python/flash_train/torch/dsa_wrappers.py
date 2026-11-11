# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSA calling-convention adapters for flash_train.torch operators."""

from __future__ import annotations

from typing import Optional

import torch

from ._flash_train_torch import (
    sparse_attn_bwd as _sparse_attn_bwd_op,
)


def _contiguous_inputs(*tensors):
    return tuple(t.contiguous() for t in tensors)


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


DSA = DSANamespace()

__all__ = ["DSA", "DSANamespace"]
