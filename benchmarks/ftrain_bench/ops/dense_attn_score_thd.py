# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""dense_attn_score_thd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.dense_attn_score_thd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_attn_full_score、TestRealKernelFusedIndexerSparseAttnThd；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,heads,head_dim) 即 (total_q,H,D)；保留 THD_SHAPE_PRESETS 的可变段长与 ratio=4，score 列数为 max_seqlen_kv。
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
    # einsum broadcasts singleton head axes; K can share one head across all Q heads.
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    values = (logits - lse.float().unsqueeze(-1)).exp().sum(2)
    valid = _torch_mask(q.shape[1], k.shape[1], q.device, ratio)[None]
    denom = values.masked_fill(~valid, 0).sum(-1)
    values = values.masked_fill(~valid, -torch.inf)
    return (values[0], denom[0]) if packed else (values, denom)


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (8192, 64, 512): {"segments": [4096, 4096], "ratio": 4},
    (12288, 16, 512): {"segments": [4096, 8192], "ratio": 4},
    (32768, 64, 512): {"segments": [16384, 16384], "ratio": 128},
    (8192, 64, 576): {"segments": [4096, 4096], "ratio": 1},
    (12288, 8, 576): {"segments": [4096, 8192], "ratio": 1},
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
    kv_tokens = cu_k[-1]
    max_k = max(length // ratio for length in segments)
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["k"] = (
        torch.rand((kv_tokens, 1, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["lse"] = (torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.float32
    )
    p["cu_seqlens_q"] = torch.tensor(cu_q, device="cuda", dtype=torch.int32)
    p["cu_seqlens_kv"] = torch.tensor(cu_k, device="cuda", dtype=torch.int32)
    p["q_causal_offsets"] = torch.zeros(
        (len(segments),), device="cuda", dtype=torch.int32
    )
    p["ratio"] = ratio
    p["softmax_scale"] = 0.125
    p["max_seqlen_kv"] = max_k
    p["_segments"] = segments
    return p


def _baseline(p):
    out = torch.full(
        (p["q"].shape[0], p["max_seqlen_kv"]), -torch.inf, device=p["q"].device
    )
    denom = torch.empty(p["q"].shape[0], device=p["q"].device)
    qs = ks = 0
    for length in p["_segments"]:
        nk = length // p["ratio"]
        score, row_denom = _torch_dense_attn(
            p["q"][qs : qs + length],
            p["k"][ks : ks + nk],
            p["lse"][qs : qs + length],
            p["softmax_scale"],
            p["ratio"],
        )
        out[qs : qs + length, :nk], denom[qs : qs + length] = score, row_denom
        qs += length
        ks += nk
    return (out, denom)


def _ftrain_dense_attn_score_thd(p):
    import flash_train.torch

    return flash_train.torch.dense_attn_score_thd(
        q=p["q"],
        k=p["k"],
        lse=p["lse"],
        cu_seqlens_q=p["cu_seqlens_q"],
        cu_seqlens_kv=p["cu_seqlens_kv"],
        ratio=p["ratio"],
        softmax_scale=p["softmax_scale"],
        max_seqlen_kv=p["max_seqlen_kv"],
        q_causal_offsets=p["q_causal_offsets"],
    )


register(
    {
        "name": "dense_attn_score_thd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.dense_attn_score_thd",
            "call": _ftrain_dense_attn_score_thd,
        },
        "suites": {
            "standard": [
                (8192, 64, 512),
                (12288, 16, 512),
                (32768, 64, 512),
                (8192, 64, 576),
                (12288, 8, 576),
            ],
            "standard_v1": [
                (8192, 64, 512),
                (12288, 16, 512),
                (32768, 64, 512),
                (8192, 64, 576),
                (12288, 8, 576),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
