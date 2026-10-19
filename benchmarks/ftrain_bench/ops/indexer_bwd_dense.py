# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""indexer_bwd_dense 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.indexer_bwd_dense（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_dense_indexer_loss、TestRealKernelDenseIndexerBackward；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,heads,head_dim) 即 (Sq,H,D)；BSHD 预设另保存 B、Sk 和压缩比；GLM-5.2 为 ratio=1。
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


def _torch_mask(rows, columns, device, ratio=1):
    return (
        torch.arange(columns, device=device)[None, :]
        < (torch.arange(rows, device=device)[:, None] + 1) // ratio
    )


def _torch_dense_indexer(q, k, w, scale, ratio=1):
    packed = q.ndim == 3
    if packed:
        q, k, w = q[None], k[None], w[None]
    # einsum broadcasts singleton head axes; K can share one head across all Q heads.
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
    scores = (logits.relu() * w.float().unsqueeze(-1)).sum(2)
    scores = scores.masked_fill(
        ~_torch_mask(q.shape[1], k.shape[1], q.device, ratio)[None], -torch.inf
    )
    lse = torch.logsumexp(scores, -1)
    return (scores[0], lse[0]) if packed else (scores, lse)


def _torch_indexer_dense_backward(p):
    packed = p["index_q"].ndim == 3
    q, w, k = (p[n].float() for n in ("index_q", "weights", "index_k"))
    teacher, norm, logits, lse = (
        p[n].float() for n in ("attn_score", "attn_l1norm", "index_score", "index_lse")
    )
    if packed:
        q, w, k, teacher, norm, logits, lse = (
            t[None] for t in (q, w, k, teacher, norm, logits, lse)
        )
    valid = _torch_mask(q.shape[1], k.shape[1], q.device, p["ratio"])[None]
    target = (teacher / (norm.unsqueeze(-1) + 1e-10)).clamp_min(math.exp(-100))
    log_p = logits - lse.unsqueeze(-1)
    direct = -(target * (log_p >= -100)).masked_fill(~valid, 0) / p.get(
        "_normalizer", q.shape[0] * q.shape[1]
    )
    ds = (direct - log_p.exp() * direct.sum(-1, keepdim=True)).masked_fill(~valid, 0)
    raw = torch.einsum("bqhd,bkd->bqhk", q, k) * p["sm_scale"]
    dw = (raw.relu() * ds.unsqueeze(2)).sum(-1)
    dg = ds.unsqueeze(2) * w.unsqueeze(-1) * (raw > 0) * p["sm_scale"]
    dq = torch.einsum("bqhk,bkd->bqhd", dg, k)
    dk = torch.einsum("bqhk,bqhd->bkd", dg, q)
    outputs = (ds, dq, dw, dk)
    if packed:
        outputs = tuple(t[0] for t in outputs)
    return (
        outputs[0],
        *(
            t.to(p[n].dtype)
            for t, n in zip(outputs[1:], ("index_q", "weights", "index_k"))
        ),
    )


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
        torch.rand((batch, a, kv_tokens), device="cuda", generator=generator) * 0.2
        - 0.1
    ).to(torch.float32)
    p["attn_l1norm"] = (
        torch.rand((batch, a), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["index_score"] = (
        torch.rand((batch, a, kv_tokens), device="cuda", generator=generator) * 0.2
        - 0.1
    ).to(torch.float32)
    p["index_lse"] = (
        torch.rand((batch, a), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["sm_scale"] = 0.25
    p["loss_coeff"] = 1
    p["grad_loss"] = 1
    p["ratio"] = config.get("ratio", 1)
    p["index_score"], p["index_lse"] = _torch_dense_indexer(
        p["index_q"],
        p["index_k"].unsqueeze(-2),
        p["weights"],
        p["sm_scale"],
        p["ratio"],
    )
    valid = _torch_mask(a, kv_tokens, p["index_q"].device, p["ratio"])
    p["attn_score"] = p["attn_score"].abs().masked_fill(~valid, 0)
    p["attn_l1norm"] = p["attn_score"].sum(-1)
    p["_initial_score"] = p["index_score"].clone()
    return p


def _baseline(p):
    # Match the binding's input copy in this operator's timing scope.
    p["index_score"].copy_(p["_initial_score"])
    return _torch_indexer_dense_backward(p)


def _ftrain_indexer_bwd_dense(p):
    import flash_train.torch

    # The backward kernel overwrites this input; restore it locally.
    p["index_score"].copy_(p["_initial_score"])
    return flash_train.torch.indexer_bwd_dense(
        index_q=p["index_q"],
        weights=p["weights"],
        index_k=p["index_k"],
        attn_score=p["attn_score"],
        attn_l1norm=p["attn_l1norm"],
        index_score=p["index_score"],
        index_lse=p["index_lse"],
        sm_scale=p["sm_scale"],
        loss_coeff=p["loss_coeff"],
        grad_loss=p["grad_loss"],
        ratio=p["ratio"],
    )


register(
    {
        "name": "indexer_bwd_dense",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (mean over B*Sq)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.indexer_bwd_dense",
            "call": _ftrain_indexer_bwd_dense,
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
