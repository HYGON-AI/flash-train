# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""indexer_topk_thd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.indexer_topk_thd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
TestRealKernelFusedIndexerSparseAttnThd、cuDNN indexer_top_k_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (total_queries,max_kv_length,top_k)；取 THD_SHAPE_PRESETS 的总 Q 行数、最大段 KV 长度和 TopK，保留分段和压缩比。
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


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (8192, 1024, 512): {"segments": [4096, 4096], "ratio": 4},
    (8192, 1024, 1024): {"segments": [4096, 4096], "ratio": 4},
    (12288, 2048, 1024): {"segments": [4096, 8192], "ratio": 4},
    (32768, 128, 128): {"segments": [16384, 16384], "ratio": 128},
    (8192, 4096, 2048): {"segments": [4096, 4096], "ratio": 1},
    (12288, 8192, 2048): {"segments": [4096, 8192], "ratio": 1},
}


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    config = _STANDARD_PARAMETERS.get(tuple(shape), {})
    segments = config.get("segments", [a])
    ratio = config.get("ratio", 1)
    cu_q = [0]
    cu_k = [0]
    for length in segments:
        cu_q.append(cu_q[-1] + length)
        cu_k.append(cu_k[-1] + length // ratio)
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["input_values"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["cu_seqlens_q"] = torch.tensor(cu_q, device="cuda", dtype=torch.int32)
    p["cu_seqlens_kv"] = torch.tensor(cu_k, device="cuda", dtype=torch.int32)
    p["q_causal_offsets"] = torch.zeros(
        (len(segments),), device="cuda", dtype=torch.int32
    )
    p["ratio"] = ratio
    p["top_k"] = c
    p["order_values"] = 1
    p["_segments"] = segments
    return p


def _baseline(p):
    values = p["input_values"].clone()
    start = 0
    columns = torch.arange(values.shape[1], device=values.device)[None]
    for length in p["_segments"]:
        limit = (torch.arange(length, device=values.device)[:, None] + 1) // p["ratio"]
        valid = (columns < limit) & (columns < length // p["ratio"])
        values[start : start + length].masked_fill_(~valid, -torch.inf)
        start += length
    selected, indices = values.topk(p["top_k"], sorted=True)
    return indices.masked_fill(torch.isneginf(selected), -1).to(torch.int32), selected


def _ftrain_indexer_topk_thd(p):
    import flash_train.torch

    return flash_train.torch.indexer_topk_thd(
        input_values=p["input_values"],
        cu_seqlens_q=p["cu_seqlens_q"],
        cu_seqlens_kv=p["cu_seqlens_kv"],
        ratio=p["ratio"],
        top_k=p["top_k"],
        order_values=p["order_values"],
        q_causal_offsets=p["q_causal_offsets"],
    )


register(
    {
        "name": "indexer_topk_thd",
        "precisions": ["fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.indexer_topk_thd",
            "call": _ftrain_indexer_topk_thd,
        },
        "suites": {
            "standard": [
                (8192, 1024, 512),
                (8192, 1024, 1024),
                (12288, 2048, 1024),
                (32768, 128, 128),
                (8192, 4096, 2048),
                (12288, 8192, 2048),
            ],
            "standard_v1": [
                (8192, 1024, 512),
                (8192, 1024, 1024),
                (12288, 2048, 1024),
                (32768, 128, 128),
                (8192, 4096, 2048),
                (12288, 8192, 2048),
            ],
        },
        "tolerances": {"fp32": (0.0002, 2e-05)},
    }
)
