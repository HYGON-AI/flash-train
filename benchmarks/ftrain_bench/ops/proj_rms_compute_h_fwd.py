# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""proj_rms_compute_h_fwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.proj_rms_compute_h_fwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_proj_rms_compute_h；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (rows,streams,reduction) 即 (M,n,K)；weight 为 (n*n+2*n,K)，输出宽度为 n、n、n*n。
形状来源：DeepSeek-V4-Flash/Pro 官方配置；N=4，隐藏维度 4096/7168，投影 K=N*hidden_size。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
eps=compute_h_eps=1e-6；r 为 RMS，alpha / bias 为 FP32。
数据和精度沿用本模块；这里只转换形状，公共计时流程不变。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, h_eps):
    proj = x.float() @ weight.float().T
    r = x.float().square().mean(-1, keepdim=True).sqrt()
    normalized = proj / (r + eps)
    pre = (normalized[:, :n] * alpha_pre.float() + bias.float()[:n]).sigmoid() + h_eps
    post = (
        2
        * (
            normalized[:, n : 2 * n] * alpha_post.float() + bias.float()[n : 2 * n]
        ).sigmoid()
    )
    res = normalized[:, 2 * n :] * alpha_res.float() + bias.float()[2 * n :]
    return pre, post, res, r, proj


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["x"] = (torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["weight"] = (
        torch.rand((b * b + 2 * b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["alpha_pre"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["alpha_post"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["alpha_res"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["bias"] = (
        torch.rand((b * b + 2 * b,), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["n"] = b
    p["eps"] = 1e-6
    p["compute_h_eps"] = 1e-6
    return p


def _baseline(p):
    return tuple(
        t.to(p["x"].dtype)
        for t in _torch_compute_h(
            p["x"],
            p["weight"],
            p["alpha_pre"],
            p["alpha_post"],
            p["alpha_res"],
            p["bias"],
            p["n"],
            p["eps"],
            p["compute_h_eps"],
        )
    )


def _ftrain_proj_rms_compute_h_fwd(p):
    import flash_train.torch

    return flash_train.torch.proj_rms_compute_h_fwd(
        x=p["x"],
        weight=p["weight"],
        alpha_pre=p["alpha_pre"],
        alpha_post=p["alpha_post"],
        alpha_res=p["alpha_res"],
        bias=p["bias"],
        n=p["n"],
        eps=p["eps"],
        compute_h_eps=p["compute_h_eps"],
    )


register(
    {
        "name": "proj_rms_compute_h_fwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.proj_rms_compute_h_fwd",
            "call": _ftrain_proj_rms_compute_h_fwd,
        },
        "suites": {
            "standard": [
                (4096, 4, 16384),
                (8192, 4, 16384),
                (16384, 4, 16384),
                (32768, 4, 16384),
                (4096, 4, 28672),
                (8192, 4, 28672),
                (16384, 4, 28672),
                (32768, 4, 28672),
            ],
            "standard_v1": [
                (4096, 4, 16384),
                (8192, 4, 16384),
                (16384, 4, 16384),
                (32768, 4, 16384),
                (4096, 4, 28672),
                (8192, 4, 28672),
                (16384, 4, 28672),
                (32768, 4, 28672),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
