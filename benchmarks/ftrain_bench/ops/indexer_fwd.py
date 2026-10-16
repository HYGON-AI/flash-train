# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""indexer_fwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.indexer_fwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_indexer_full_score、cuDNN indexer_forward_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。

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


def _torch_dense_indexer(q, k, w, scale, ratio=1):
    packed = q.ndim == 3
    if packed:
        q, k, w = q[None], k[None], w[None]
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    scores = (logits.relu() * w.float().unsqueeze(-1)).sum(2)
    scores = scores.masked_fill(
        ~_torch_mask(q.shape[1], k.shape[1], q.device, ratio)[None], -torch.inf
    )
    lse = torch.logsumexp(scores, -1)
    return (scores[0], lse[0]) if packed else (scores, lse)


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 64, 128): {"batch": 1, "kv_tokens": 1024, "ratio": 4},
    (8192, 64, 128): {"batch": 1, "kv_tokens": 2048, "ratio": 4},
    (16384, 64, 128): {"batch": 1, "kv_tokens": 128, "ratio": 128},
    (4096, 8, 128): {"batch": 2, "kv_tokens": 1024, "ratio": 4},
    (4096, 32, 128): {"batch": 1, "kv_tokens": 4096, "ratio": 1},
    (8192, 4, 128): {"batch": 1, "kv_tokens": 8192, "ratio": 1},
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
    p["w"] = (
        torch.rand((batch, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["ratio"] = config.get("ratio", 1)
    p["sm_scale"] = 0.25
    return p


def _baseline(p):
    return _torch_dense_indexer(p["q"], p["k"], p["w"], p["sm_scale"], p["ratio"])[0]


def _ftrain_indexer_fwd(p):
    import flash_train.torch

    return flash_train.torch.indexer_fwd(
        q=p["q"], k=p["k"], w=p["w"], ratio=p["ratio"], sm_scale=p["sm_scale"]
    )


register(
    {
        "name": "indexer_fwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.indexer_fwd",
            "call": _ftrain_indexer_fwd,
        },
        "suites": {
            "standard": [
                (4096, 64, 128),
                (8192, 64, 128),
                (16384, 64, 128),
                (4096, 8, 128),
                (4096, 32, 128),
                (8192, 4, 128),
            ],
            "standard_v1": [
                (4096, 64, 128),
                (8192, 64, 128),
                (16384, 64, 128),
                (4096, 8, 128),
                (4096, 32, 128),
                (8192, 4, 128),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
