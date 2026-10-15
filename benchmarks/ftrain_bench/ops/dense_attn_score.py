# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""dense_attn_score 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.dense_attn_score（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_attn_full_score、cuDNN dense_attn_score_recompute_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,heads,head_dim) 即 (Sq,H,D)；BSHD 预设另保存 B、Sk 和压缩比；GLM-5.2 为 ratio=1。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_mask(rows, columns, device, ratio=1):
    return (
        torch.arange(columns, device=device)[None, :]
        < (torch.arange(rows, device=device)[:, None] + 1) // ratio
    )


def _torch_dense_attn(q, k, lse, scale, ratio=1):
    packed = q.ndim == 3
    if packed:
        q, k, lse = q[None], k[None], lse[None]
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    values = (logits - lse.float().unsqueeze(-1)).exp().sum(2)
    valid = _torch_mask(q.shape[1], k.shape[1], q.device, ratio)[None]
    denom = values.masked_fill(~valid, 0).sum(-1)
    values = values.masked_fill(~valid, -torch.inf)
    return (values[0], denom[0]) if packed else (values, denom)


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 64, 512): {"batch": 1, "kv_tokens": 1024, "ratio": 4},
    (4096, 128, 512): {"batch": 1, "kv_tokens": 1024, "ratio": 4},
    (8192, 8, 512): {"batch": 2, "kv_tokens": 2048, "ratio": 4},
    (8192, 16, 512): {"batch": 1, "kv_tokens": 2048, "ratio": 4},
    (16384, 64, 512): {"batch": 1, "kv_tokens": 128, "ratio": 128},
    (4096, 64, 576): {"batch": 1, "kv_tokens": 4096, "ratio": 1},
    (8192, 8, 576): {"batch": 1, "kv_tokens": 8192, "ratio": 1},
}


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    config = _STANDARD_PARAMETERS.get(tuple(shape), {})
    batch = config.get("batch", 1)
    kv_tokens = config.get("kv_tokens", a)
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (
        torch.rand((batch, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["k"] = (
        torch.rand((batch, kv_tokens, 1, c), device="cuda", generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    p["lse"] = (
        torch.rand((batch, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["ratio"] = config.get("ratio", 1)
    p["softmax_scale"] = 0.125
    return p


def _baseline(p):
    return _torch_dense_attn(p["q"], p["k"], p["lse"], p["softmax_scale"], p["ratio"])


def _ftrain_dense_attn_score(p):
    import flash_train.torch

    return flash_train.torch.dense_attn_score(
        q=p["q"],
        k=p["k"],
        lse=p["lse"],
        ratio=p["ratio"],
        softmax_scale=p["softmax_scale"],
    )


register(
    {
        "name": "dense_attn_score",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.dense_attn_score",
            "call": _ftrain_dense_attn_score,
        },
        "suites": {
            "standard": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
                (16384, 64, 512),
                (4096, 64, 576),
                (8192, 8, 576),
            ],
            "standard_v1": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
                (16384, 64, 512),
                (4096, 64, 576),
                (8192, 8, 576),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
