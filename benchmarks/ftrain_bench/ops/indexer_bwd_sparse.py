# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""indexer_bwd_sparse 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.indexer_bwd_sparse（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_indexer_predict_sparse / _ref_attn_target_sparse 的 KL 梯度；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (query_tokens,head_dim,top_k) 即 (Sq,D,K)；本地预设保留 B、H 和 Sk，避免不同 TopK 的 DSv4 配置混为同一形状。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
KL 梯度按全部查询行 B*Sq 归一化，与源参考和被测内核一致。
L1 原地输出前在算子内恢复输入，复制计入双方耗时；C 侧按算子的原地约束复用输入分数缓冲区。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import math

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


def _torch_indexer_sparse_backward(p):
    q, w, k = (p[n].float() for n in ("index_q", "weights", "index_k"))
    keys, raw = _torch_gathered(q, k, p["topk_indices"], p["sm_scale"])
    target, probability = p["attn_score"].float(), p["index_score"].float()
    direct = (
        -target.clamp_min(math.exp(-100))
        * (probability >= math.exp(-100))
        / (q.shape[0] * q.shape[1])
    )
    ds = direct - probability * direct.sum(-1, keepdim=True)
    dw = (raw.relu() * ds.unsqueeze(2)).sum(-1)
    dg = ds.unsqueeze(2) * w.unsqueeze(-1) * (raw > 0) * p["sm_scale"]
    dq = torch.einsum("bqhk,bqkd->bqhd", dg, keys)
    contribution = torch.einsum("bqhk,bqhd->bqkd", dg, q)
    dk = torch.zeros_like(k)
    for b in range(q.shape[0]):
        dk[b].index_add_(
            0, p["topk_indices"][b].flatten().long(), contribution[b].flatten(0, 1)
        )
    return (
        ds,
        dq.to(p["index_q"].dtype),
        dw.to(p["weights"].dtype),
        dk.to(p["index_k"].dtype),
    )


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
    p["index_q"] = (
        torch.rand((batch, a, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["weights"] = (
        torch.rand((batch, a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["index_k"] = (
        torch.rand((batch, kv_tokens, c), device="cuda", generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    p["attn_score"] = (
        torch.rand((batch, a, top_k), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["index_score"] = (
        torch.rand((batch, a, top_k), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["topk_indices"] = (
        (torch.arange(top_k, device="cuda", dtype=torch.int32) % kv_tokens)
        .expand((batch, a, top_k))
        .contiguous()
    )
    p["sm_scale"] = 0.25
    p["loss_coeff"] = 1
    p["grad_loss"] = 1
    p["topk_indices_global"] = 0
    p["attn_score"] = p["attn_score"].abs()
    p["index_score"] = _torch_sparse_predict(
        p["index_q"], p["index_k"], p["weights"], p["topk_indices"], p["sm_scale"]
    )
    p["_initial_score"] = p["attn_score"].clone()
    return p


def _baseline(p):
    # Match the binding's input copy in this operator's timing scope.
    p["attn_score"].copy_(p["_initial_score"])
    return _torch_indexer_sparse_backward(p)


def _ftrain_indexer_bwd_sparse(p):
    import flash_train.torch

    # The backward kernel overwrites this input; restore it locally.
    p["attn_score"].copy_(p["_initial_score"])
    return flash_train.torch.indexer_bwd_sparse(
        index_q=p["index_q"],
        weights=p["weights"],
        index_k=p["index_k"],
        attn_score=p["attn_score"],
        index_score=p["index_score"],
        topk_idxs=p["topk_indices"],
        sm_scale=p["sm_scale"],
        loss_coeff=p["loss_coeff"],
        grad_loss=p["grad_loss"],
        topk_indices_global=p["topk_indices_global"],
    )


register(
    {
        "name": "indexer_bwd_sparse",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (mean over B*Sq)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.indexer_bwd_sparse",
            "call": _ftrain_indexer_bwd_sparse,
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
