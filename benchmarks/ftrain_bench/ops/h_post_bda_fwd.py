# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""h_post_bda_fwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.h_post_bda_fwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_h_post_bda；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,streams,width) 即 (T,N,C)，T=S*B；源脚本 (S,B,N,C) 展平为 (T,1,N,C)，保留全部行和通道。
形状来源：DeepSeek-V4-Flash/Pro 官方配置；N=4，隐藏维度 4096/7168，投影 K=N*hidden_size。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
数据和精度沿用本模块；这里只转换形状，公共计时流程不变。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_post_bda(h_res, residual, h_post, x, bias):
    return torch.einsum(
        "...ij,...id->...jd", h_res.float(), residual.float()
    ) + h_post.float().unsqueeze(-1) * (x.float() + bias.float()).unsqueeze(-2)


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["h_res"] = (
        torch.rand((a, 1, b, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["residual"] = (
        torch.rand((a, 1, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["h_post"] = (
        torch.rand((a, 1, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["x"] = (torch.rand((a, 1, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        dtype
    )
    p["bias"] = (torch.rand((c,), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.float32
    )
    return p


def _baseline(p):
    return _torch_post_bda(
        p["h_res"], p["residual"], p["h_post"], p["x"], p["bias"]
    ).to(p["x"].dtype)


def _ftrain_h_post_bda_fwd(p):
    import flash_train.torch

    return flash_train.torch.h_post_bda_fwd(
        h_res=p["h_res"],
        residual=p["residual"],
        h_post=p["h_post"],
        x=p["x"],
        bias=p["bias"],
    )


register(
    {
        "name": "h_post_bda_fwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.h_post_bda_fwd",
            "call": _ftrain_h_post_bda_fwd,
        },
        "suites": {
            "standard": [
                (4096, 4, 4096),
                (8192, 4, 4096),
                (16384, 4, 4096),
                (32768, 4, 4096),
                (4096, 4, 7168),
                (8192, 4, 7168),
                (16384, 4, 7168),
                (32768, 4, 7168),
            ],
            "standard_v1": [
                (4096, 4, 4096),
                (8192, 4, 4096),
                (16384, 4, 4096),
                (32768, 4, 4096),
                (4096, 4, 7168),
                (8192, 4, 7168),
                (16384, 4, 7168),
                (32768, 4, 7168),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
