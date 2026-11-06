# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""h_aggregate_bwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.h_aggregate_bwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_h_aggregate 的 autograd 梯度；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (tokens,streams,width) 即 (T,N,C)，T=S*B；源脚本 (S,B,N,C) 展平为 (T,1,N,C)，保留全部行和通道。
形状来源：DeepSeek-V4-Flash/Pro 官方配置；N=4，隐藏维度 4096/7168，投影 K=N*hidden_size。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
数据和精度沿用本模块；这里只转换形状，公共计时流程不变。
"""

import torch
from . import register

DTYPES = {"fp32": torch.float32}


def _torch_aggregate_backward(p):
    grad, x, h = (p[n].float() for n in ("grad_aggregated", "x", "h_pre"))
    return (grad.unsqueeze(-2) * h.unsqueeze(-1)).to(p["x"].dtype), (
        grad.unsqueeze(-2) * x
    ).sum(-1)


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["grad_aggregated"] = (
        torch.rand((a, 1, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["x"] = (
        torch.rand((a, 1, b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["h_pre"] = (
        torch.rand((a, 1, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    return p


def _baseline(p):
    return _torch_aggregate_backward(p)


def _ftrain_h_aggregate_bwd(p):
    import flash_train.torch

    return flash_train.torch.h_aggregate_bwd(
        grad_aggregated=p["grad_aggregated"], x=p["x"], h_pre=p["h_pre"]
    )


register(
    {
        "name": "h_aggregate_bwd",
        "precisions": ["fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.h_aggregate_bwd",
            "call": _ftrain_h_aggregate_bwd,
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
        "tolerances": {"fp32": (0.0002, 2e-05)},
    }
)
