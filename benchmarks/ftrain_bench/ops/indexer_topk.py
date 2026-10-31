# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""indexer_topk 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.indexer_topk（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
TestRealKernelIndexerTopk、cuDNN indexer_top_k_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (rows,columns,top_k) 直接取 TOPK_PRESETS；FP32 数值，INT32 索引，order_values=True。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"fp32": torch.float32}


def _torch_mask(rows, columns, device):
    return (
        torch.arange(columns, device=device)[None, :]
        <= torch.arange(rows, device=device)[:, None]
    )


def _torch_topk(p, packed=False):
    values = p["input_values"]
    if packed:
        valid = _torch_mask(values.shape[0], values.shape[1], values.device)
        values = values.masked_fill(~valid, -torch.inf)
    selected, indices = values.topk(p["top_k"], sorted=True)
    if packed:
        indices = indices.masked_fill(torch.isneginf(selected), -1)
    return indices.to(torch.int32), selected


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["input_values"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["seq_lens"] = torch.full((a,), b, device="cuda", dtype=torch.int32)
    p["top_k"] = c
    p["next_n"] = 1
    p["order_values"] = 1
    return p


def _baseline(p):
    return _torch_topk(p)


def _ftrain_indexer_topk(p):
    import flash_train.torch

    return flash_train.torch.indexer_topk(
        input_values=p["input_values"],
        seq_lens=p["seq_lens"],
        top_k=p["top_k"],
        next_n=p["next_n"],
        order_values=p["order_values"],
    )


register(
    {
        "name": "indexer_topk",
        "precisions": ["fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.indexer_topk",
            "call": _ftrain_indexer_topk,
        },
        "suites": {
            "standard": [
                (4096, 1024, 512),
                (4096, 1024, 1024),
                (8192, 2048, 512),
                (8192, 2048, 1024),
                (16384, 128, 128),
                (4096, 4096, 2048),
                (8192, 8192, 2048),
            ],
            "standard_v1": [
                (4096, 1024, 512),
                (4096, 1024, 1024),
                (8192, 2048, 512),
                (8192, 2048, 1024),
                (16384, 128, 128),
                (4096, 4096, 2048),
                (8192, 8192, 2048),
            ],
        },
        "tolerances": {"fp32": (0.0002, 2e-05)},
    }
)
