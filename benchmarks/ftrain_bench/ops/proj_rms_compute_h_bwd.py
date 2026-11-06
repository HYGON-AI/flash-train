# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""proj_rms_compute_h_bwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.proj_rms_compute_h_bwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_proj_rms_compute_h 的 autograd 梯度；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。
前向与计算图在 make_args 中一次性构建，不计时；基线仅重复执行反向及输出类型转换。

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


def _torch_compute_h_backward(p):
    gx, gw, ga, gb, gc, g_bias = torch.autograd.grad(
        p["_torch_outputs"], p["_torch_inputs"], p["_torch_incoming"], retain_graph=True
    )
    return (
        gx.to(p["x"].dtype),
        gw.to(p["weight"].dtype),
        torch.cat((ga, gb, gc)),
        g_bias,
    )


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["grad_h_pre"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["grad_h_post"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["grad_h_res"] = (
        torch.rand((a, b * b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["grad_r"] = (
        torch.rand((a, 1), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["h_pre"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["h_post"] = (
        torch.rand((a, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["proj"] = (
        torch.rand((a, b * b + 2 * b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["r"] = (torch.rand((a, 1), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["x"] = (torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["weight"] = (
        torch.rand((b * b + 2 * b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["alpha_pre"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["alpha_post"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["alpha_res"] = torch.ones((1,), device="cuda", dtype=torch.float32)
    p["n"] = b
    p["eps"] = 1e-6
    p["compute_h_eps"] = 1e-6
    names = ("x", "weight", "alpha_pre", "alpha_post", "alpha_res")
    values = tuple(p[n].detach().float().requires_grad_() for n in names)
    bias = torch.zeros(b * b + 2 * b, device="cuda", requires_grad=True)
    p["_torch_inputs"] = (*values, bias)
    hp, hpost, hres, r, proj = _torch_compute_h(
        *p["_torch_inputs"], b, p["eps"], p["compute_h_eps"]
    )
    p["_torch_outputs"] = (hp, hpost, hres, r)
    p["_torch_incoming"] = tuple(
        p[n].float() for n in ("grad_h_pre", "grad_h_post", "grad_h_res", "grad_r")
    )
    p["h_pre"], p["h_post"], p["r"], p["proj"] = (
        t.detach().to(p["x"].dtype) for t in (hp, hpost, r, proj)
    )
    return p


def _baseline(p):
    return _torch_compute_h_backward(p)


def _ftrain_proj_rms_compute_h_bwd(p):
    import flash_train.torch

    return flash_train.torch.proj_rms_compute_h_bwd(
        grad_h_pre=p["grad_h_pre"],
        grad_h_post=p["grad_h_post"],
        grad_h_res=p["grad_h_res"],
        grad_r=p["grad_r"],
        h_pre=p["h_pre"],
        h_post=p["h_post"],
        proj=p["proj"],
        r=p["r"],
        x=p["x"],
        weight=p["weight"],
        alpha_pre=p["alpha_pre"],
        alpha_post=p["alpha_post"],
        alpha_res=p["alpha_res"],
        n=p["n"],
        eps=p["eps"],
        compute_h_eps=p["compute_h_eps"],
    )


register(
    {
        "name": "proj_rms_compute_h_bwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (forward excluded)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.proj_rms_compute_h_bwd",
            "call": _ftrain_proj_rms_compute_h_bwd,
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
