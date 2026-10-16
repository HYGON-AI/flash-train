# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""compactify 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.compactify（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
build_flat_topk_idxs 的 compact 分支、cuDNN compactify_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (rows,columns,invalid_stride)；前两维取 COMPACTIFY_PRESETS，invalid_stride=3，INT32 -1 表示无效项。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"int32": torch.int32}


def _torch_compact(indices):
    positions = torch.arange(indices.shape[1], device=indices.device).expand_as(indices)
    order = positions.masked_fill(indices < 0, indices.shape[1]).argsort(
        dim=-1, stable=True
    )
    output = indices.gather(-1, order)
    length = (indices >= 0).sum(-1).to(torch.int32)
    return output, length


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    p = {}
    p["indices"] = (
        torch.arange(b, device="cuda", dtype=torch.int32).expand(a, b).contiguous()
    )
    p["indices"][:, ::c] = -1
    return p


def _baseline(p):
    return _torch_compact(p["indices"])


def _ftrain_compactify(p):
    import flash_train.torch

    return flash_train.torch.compactify(indices=p["indices"])


register(
    {
        "name": "compactify",
        "precisions": ["int32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {"name": "flash_train.torch.compactify", "call": _ftrain_compactify},
        "suites": {
            "standard": [
                (4096, 640, 3),
                (4096, 1152, 3),
                (8192, 640, 3),
                (8192, 1152, 3),
                (16384, 256, 3),
                (4096, 2048, 3),
                (8192, 2048, 3),
            ],
            "standard_v1": [
                (4096, 640, 3),
                (4096, 1152, 3),
                (8192, 640, 3),
                (8192, 1152, 3),
                (16384, 256, 3),
                (4096, 2048, 3),
                (8192, 2048, 3),
            ],
        },
        "tolerances": {"int32": (0, 0)},
    }
)
