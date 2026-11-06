# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sparse_attn_fwd_indexer 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.sparse_attn_fwd_indexer（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_dsa_fwd_flash_mla 的 indexer_topk 分支、FlashMLA flash_mla_sparse_fwd；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,heads,head_dim) 即 (T,H,D)；本地预设另保存 KV 长度和 TopK，Q/K 使用 D=512/576，V 宽度为 512。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_sparse_attention(q, kv, sink, indices, scale):
    # Bound the temporary gather and its backward gradient on 64 GiB devices.
    outputs, normalizers, scores = [], [], []
    for begin in range(0, q.shape[0], 1024):
        queries = q[begin : begin + 1024].float()
        keys = kv.float()[indices[begin : begin + 1024].long()]
        logits = torch.einsum("thd,tkd->thk", queries, keys) * scale
        lse = torch.logsumexp(logits, -1)
        weights = torch.cat(
            (logits, sink.float()[None, :, None].expand(queries.shape[0], -1, -1)), -1
        ).softmax(-1)[..., :-1]
        outputs.append(torch.einsum("thk,tkd->thd", weights, keys[..., :512]))
        normalizers.append(lse)
        scores.append(logits)
    return torch.cat(outputs), torch.cat(normalizers), torch.cat(scores)


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 64, 512): {"kv_tokens": 5120, "top_k": 640, "indexer_topk": 512},
    (4096, 128, 512): {"kv_tokens": 5120, "top_k": 1152, "indexer_topk": 1024},
    (8192, 8, 512): {"kv_tokens": 10240, "top_k": 640, "indexer_topk": 512},
    (8192, 16, 512): {"kv_tokens": 10240, "top_k": 1152, "indexer_topk": 1024},
    (4096, 64, 576): {"kv_tokens": 4096, "top_k": 2048, "indexer_topk": 2048},
    (8192, 8, 576): {"kv_tokens": 8192, "top_k": 2048, "indexer_topk": 2048},
}


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    config = _STANDARD_PARAMETERS.get(tuple(shape), {})
    kv_tokens = config.get("kv_tokens", a)
    top_k = config.get("top_k", 128)
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["q"] = (torch.rand((a, b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["kv"] = (
        torch.rand((kv_tokens, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["attn_sink"] = (
        torch.rand((b,), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["topk_indices"] = (
        (torch.arange(top_k, device="cuda", dtype=torch.int32) % kv_tokens)
        .expand((a, top_k))
        .contiguous()
    )
    p["topk_length"] = torch.full((a,), top_k, device="cuda", dtype=torch.int32)
    p["softmax_scale"] = 0.125
    p["indexer_topk"] = config.get("indexer_topk", 32)
    return p


def _baseline(p):
    values = _torch_sparse_attention(
        p["q"], p["kv"], p["attn_sink"], p["topk_indices"], p["softmax_scale"]
    )
    return (
        values[0].to(p["q"].dtype),
        values[2].amax(-1),
        values[1],
        torch.logaddexp(
            torch.logsumexp(values[2][..., : p["indexer_topk"]], -1),
            p["attn_sink"][None],
        ),
    )


def _ftrain_sparse_attn_fwd_indexer(p):
    import flash_train.torch

    return flash_train.torch.sparse_attn_fwd_indexer(
        q=p["q"],
        kv=p["kv"],
        attn_sink=p["attn_sink"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
        indexer_topk=p["indexer_topk"],
        topk_length=p["topk_length"],
    )


register(
    {
        "name": "sparse_attn_fwd_indexer",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.sparse_attn_fwd_indexer",
            "call": _ftrain_sparse_attn_fwd_indexer,
        },
        "suites": {
            "standard": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
                (4096, 64, 576),
                (8192, 8, 576),
            ],
            "standard_v1": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
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
