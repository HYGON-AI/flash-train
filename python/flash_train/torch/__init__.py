# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash-train PyTorch bindings."""

import ctypes
import os

# The extension NEEDs libtorch.so, which only enters the process with torch.
import torch

# Wheels built with FTRAIN_BUILD_SHARED_LIBS=ON ship libflash_train.so in
# the package root; loading it by absolute path pins the bundled copy over
# anything on LD_LIBRARY_PATH. Static wheels ship no file and skip this.
_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_native_library = os.path.join(_package_root, "libflash_train.so")
if os.path.exists(_native_library):
    ctypes.CDLL(_native_library)

try:
    from ._flash_train_torch import (
        gemm,
        compactify,
        csa_compressor_bwd,
        csa_compressor_fwd,
        dense_attn_score,
        dense_attn_score_thd,
        dense_indexer_score,
        dense_indexer_score_thd,
        h_aggregate_bwd,
        h_aggregate_fwd,
        h_post_bda_bwd,
        h_post_bda_fwd,
        indexer_bwd_dense,
        indexer_bwd_dense_thd,
        indexer_bwd_sparse,
        indexer_fwd,
        indexer_fwd_thd,
        indexer_topk,
        indexer_topk_thd,
        proj_rms_bwd,
        proj_rms_compute_h_bwd,
        proj_rms_compute_h_fwd,
        proj_rms_fwd,
        sinkhorn_bwd,
        sinkhorn_fwd,
        sparse_attn_bwd,
        sparse_attn_fwd,
        sparse_attn_fwd_indexer,
        sparse_attn_target,
        sparse_indexer_predict,
    )
except ModuleNotFoundError as exc:
    # This shell ships even in wheels built without the binding; a missing
    # extension module is a build problem, not a missing dependency.
    if exc.name != "flash_train.torch._flash_train_torch":
        raise
    raise RuntimeError(
        "this wheel was built without the torch binding; rebuild it with "
        "FTRAIN_BUILD_TORCH_BINDINGS=ON"
    ) from exc

from .dsa_wrappers import DSA, DSANamespace, sparse_attn_forward
from .mhc_wrappers import (
    FusedHAggregate,
    FusedHPostBDA,
    FusedProjRms,
    FusedProjRmsComputeH,
    FusedSinkhorn,
    fused_h_aggregate,
    fused_h_post_bda,
    fused_proj_rms,
    fused_proj_rms_compute_h,
    fused_sinkhorn,
)

__all__ = [
    "gemm",
    "compactify",
    "csa_compressor_bwd",
    "csa_compressor_fwd",
    "dense_attn_score",
    "dense_attn_score_thd",
    "dense_indexer_score",
    "dense_indexer_score_thd",
    "h_aggregate_bwd",
    "h_aggregate_fwd",
    "h_post_bda_bwd",
    "h_post_bda_fwd",
    "indexer_bwd_dense",
    "indexer_bwd_dense_thd",
    "indexer_bwd_sparse",
    "indexer_fwd",
    "indexer_fwd_thd",
    "indexer_topk",
    "indexer_topk_thd",
    "proj_rms_bwd",
    "proj_rms_compute_h_bwd",
    "proj_rms_compute_h_fwd",
    "proj_rms_fwd",
    "sinkhorn_bwd",
    "sinkhorn_fwd",
    "sparse_attn_bwd",
    "sparse_attn_fwd",
    "sparse_attn_fwd_indexer",
    "sparse_attn_target",
    "sparse_indexer_predict",
    "DSA",
    "DSANamespace",
    "sparse_attn_forward",
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
]
