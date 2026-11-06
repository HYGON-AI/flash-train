# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sparse_attn_bwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.sparse_attn_bwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/transformer/experimental_attention_variant/test_dsa_kernels.py
_ref_sparse_attn_forward 的梯度、cuDNN sparse_attention_backward_wrapper；上述 CUDA/HIP 后端未纳入当前计时基线。
前向与计算图在 make_args 中一次性构建，不计时；基线仅重复执行反向及输出类型转换。

形状元组 (tokens,heads,head_dim) 即 (T,H,D)；本地预设另保存 KV 长度和 TopK，Q/K 使用 D=512/576，V 宽度为 512。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
D=576、V=512 的反向当前仅 BF16 支持，单独使用 suite=bf16_d576_v1、precision=bf16。
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


def _torch_sparse_attention_backward(p):
    grads = torch.autograd.grad(
        p["_torch_output"], p["_torch_inputs"], p["_torch_dout"], retain_graph=True
    )
    return tuple(t.to(p[n].dtype) for t, n in zip(grads, ("q", "kv", "attn_sink")))


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 64, 512): {"kv_tokens": 5120, "top_k": 640},
    (4096, 128, 512): {"kv_tokens": 5120, "top_k": 1152},
    (8192, 8, 512): {"kv_tokens": 10240, "top_k": 640},
    (8192, 16, 512): {"kv_tokens": 10240, "top_k": 1152},
    (16384, 8, 512): {"kv_tokens": 16512, "top_k": 256},
    (4096, 64, 576): {"kv_tokens": 4096, "top_k": 2048},
    (4096, 8, 576): {"kv_tokens": 4096, "top_k": 2048},
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
    p["_torch_inputs"] = tuple(
        p[n].detach().float().requires_grad_() for n in ("q", "kv", "attn_sink")
    )
    out, lse, _ = _torch_sparse_attention(
        *p["_torch_inputs"], p["topk_indices"], p["softmax_scale"]
    )
    p["_torch_output"] = out
    p["out"] = out.detach().to(p["q"].dtype)
    # Q/K may have 576 channels; V and the actual forward output still have 512.
    p["dout"] = (
        torch.rand(out.shape, device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["_torch_dout"] = p["dout"].float()
    p["lse"] = lse.detach()
    return p


def _baseline(p):
    return _torch_sparse_attention_backward(p)


def _ftrain_sparse_attn_bwd(p):
    import flash_train.torch

    return flash_train.torch.sparse_attn_bwd(
        q=p["q"],
        kv=p["kv"],
        out=p["out"],
        dout=p["dout"],
        lse=p["lse"],
        attn_sink=p["attn_sink"],
        topk_idxs=p["topk_indices"],
        softmax_scale=p["softmax_scale"],
        topk_length=p["topk_length"],
    )


register(
    {
        "name": "sparse_attn_bwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (forward excluded)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.sparse_attn_bwd",
            "call": _ftrain_sparse_attn_bwd,
        },
        "suites": {
            "standard": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
                (16384, 8, 512),
            ],
            "standard_v1": [
                (4096, 64, 512),
                (4096, 128, 512),
                (8192, 8, 512),
                (8192, 16, 512),
                (16384, 8, 512),
            ],
            "bf16_d576": [(4096, 64, 576), (4096, 8, 576)],
            "bf16_d576_v1": [(4096, 64, 576), (4096, 8, 576)],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
