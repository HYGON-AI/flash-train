# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sparse_indexer_predict 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.sparse_indexer_predict（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_indexer_predict_sparse、cuDNN sparse_indexer_score_recompute_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (query_tokens,head_dim,top_k) 即 (Sq,D,K)；本地预设保留 B、H 和 Sk，避免不同 TopK 的 DSv4 配置混为同一形状。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_gathered(q, k, indices, scale):
    # Index a 2-D matrix per batch: this HCU Torch build mis-shapes a
    # multidimensional fancy index against a batched 3-D matrix.
    keys = torch.stack([k[b].float()[indices[b].long()] for b in range(q.shape[0])])
    logits = torch.einsum("bqhd,bqkd->bqhk", q.float(), keys) * scale
    return keys, logits


def _torch_sparse_predict(q, k, w, indices, scale):
    _, logits = _torch_gathered(q, k, indices, scale)
    return (logits.relu() * w.float().unsqueeze(-1)).sum(2).softmax(-1)


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 128, 512): {
        "batch": 1,
        "kv_tokens": 1024,
        "ratio": 4,
        "heads": 64,
        "top_k": 512,
    },
    (4096, 128, 1024): {
        "batch": 1,
        "kv_tokens": 1024,
        "ratio": 4,
        "heads": 64,
        "top_k": 1024,
    },
    (8192, 128, 512): {
        "batch": 1,
        "kv_tokens": 2048,
        "ratio": 4,
        "heads": 8,
        "top_k": 512,
    },
    (16384, 128, 128): {
        "batch": 1,
        "kv_tokens": 128,
        "ratio": 128,
        "heads": 64,
        "top_k": 128,
    },
    (4096, 128, 2048): {
        "batch": 1,
        "kv_tokens": 4096,
        "ratio": 1,
        "heads": 32,
        "top_k": 2048,
    },
    (8192, 128, 2048): {
        "batch": 1,
        "kv_tokens": 8192,
        "ratio": 1,
        "heads": 4,
        "top_k": 2048,
    },
}


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    config = _STANDARD_PARAMETERS.get(tuple(shape), {})
    a, b, c = shape[0], config.get("heads", 64), shape[1]
    batch = config.get("batch", 1)
    kv_tokens = config.get("kv_tokens", max(a // 4, 1))
    top_k = config.get("top_k", shape[2])
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (
        torch.rand((batch, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["k"] = (
        torch.rand((batch, kv_tokens, c), device="cuda", generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    p["w"] = (
        torch.rand((batch, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["topk_indices"] = (
        (torch.arange(top_k, device="cuda", dtype=torch.int32) % kv_tokens)
        .expand((batch, a, top_k))
        .contiguous()
    )
    p["sm_scale"] = 0.25
    p["topk_indices_global"] = 0
    return p


def _baseline(p):
    return _torch_sparse_predict(
        p["q"], p["k"], p["w"], p["topk_indices"], p["sm_scale"]
    )


def _ftrain_sparse_indexer_predict(p):
    import flash_train.torch

    return flash_train.torch.sparse_indexer_predict(
        q=p["q"],
        k=p["k"],
        w=p["w"],
        topk_idxs=p["topk_indices"],
        sm_scale=p["sm_scale"],
        topk_indices_global=p["topk_indices_global"],
    )


register(
    {
        "name": "sparse_indexer_predict",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.sparse_indexer_predict",
            "call": _ftrain_sparse_indexer_predict,
        },
        "suites": {
            "standard": [
                (4096, 128, 512),
                (4096, 128, 1024),
                (8192, 128, 512),
                (16384, 128, 128),
                (4096, 128, 2048),
                (8192, 128, 2048),
            ],
            "standard_v1": [
                (4096, 128, 512),
                (4096, 128, 1024),
                (8192, 128, 512),
                (16384, 128, 128),
                (4096, 128, 2048),
                (8192, 128, 2048),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
