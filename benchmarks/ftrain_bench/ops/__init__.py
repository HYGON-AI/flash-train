# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""算子基准注册表：每个算子声明基线调用、被测调用与标准形状集。

被测与基线共用同一批张量、同一计时路径；基线一律为 PyTorch 组合实现
（即仓库交付的参考实现口径），不引入其他对照物。
"""

_REGISTRY = {}


def register(defn):
    _REGISTRY[defn["name"]] = defn
    return defn


def get_op(name):
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown op: {name} (available: {', '.join(sorted(_REGISTRY))})"
        )
    return _REGISTRY[name]


def registry():
    return dict(_REGISTRY)


from . import gemm  # noqa: E402,F401  置于文件底部：register 定义执行后再导入注册

from . import compactify  # noqa: E402,F401
from . import csa_compressor_bwd  # noqa: E402,F401
from . import csa_compressor_fwd  # noqa: E402,F401
from . import dense_attn_score  # noqa: E402,F401
from . import dense_attn_score_thd  # noqa: E402,F401
from . import dense_indexer_score  # noqa: E402,F401
from . import dense_indexer_score_thd  # noqa: E402,F401
from . import h_aggregate_bwd  # noqa: E402,F401
from . import h_aggregate_fwd  # noqa: E402,F401
from . import h_post_bda_bwd  # noqa: E402,F401
from . import h_post_bda_fwd  # noqa: E402,F401
from . import indexer_bwd_dense  # noqa: E402,F401
from . import indexer_bwd_dense_thd  # noqa: E402,F401
from . import indexer_bwd_sparse  # noqa: E402,F401
from . import indexer_fwd  # noqa: E402,F401
from . import indexer_fwd_thd  # noqa: E402,F401
from . import indexer_topk  # noqa: E402,F401
from . import indexer_topk_thd  # noqa: E402,F401
from . import proj_rms_bwd  # noqa: E402,F401
from . import proj_rms_compute_h_bwd  # noqa: E402,F401
from . import proj_rms_compute_h_fwd  # noqa: E402,F401
from . import proj_rms_fwd  # noqa: E402,F401
from . import sinkhorn_bwd  # noqa: E402,F401
from . import sinkhorn_fwd  # noqa: E402,F401
from . import sparse_attn_bwd  # noqa: E402,F401
from . import sparse_attn_fwd  # noqa: E402,F401
from . import sparse_attn_fwd_indexer  # noqa: E402,F401
from . import sparse_attn_target  # noqa: E402,F401
from . import sparse_indexer_predict  # noqa: E402,F401
