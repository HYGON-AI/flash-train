# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
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
