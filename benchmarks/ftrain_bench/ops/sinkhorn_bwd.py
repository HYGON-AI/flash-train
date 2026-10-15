# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sinkhorn_bwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.sinkhorn_bwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_sinkhorn 的 autograd 梯度；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。
前向与计算图在 make_args 中一次性构建，不计时；基线仅重复执行反向及输出类型转换。

形状元组 (batch,matrix_size,iterations) 即 (S*B,N,I)；源脚本 (S,B,N,I) 展平为 (S*B,N,N)，I 为归一化迭代次数。
形状来源：DeepSeek-V4-Flash/Pro 官方配置；N=4，隐藏维度 4096/7168，投影 K=N*hidden_size。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
eps=1e-6，m_init 使用 FP32。
数据和精度沿用本模块；这里只转换形状，公共计时流程不变。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_normalize(state, iterations, eps):
    for _ in range(iterations):
        state = state / (state.sum(-1, keepdim=True) + eps)
        state = state / (state.sum(-2, keepdim=True) + eps)
    return state


def _torch_sinkhorn_backward(p):
    grad = torch.autograd.grad(
        p["_torch_output"], p["_torch_state"], p["_torch_incoming"], retain_graph=True
    )[0]
    softmax = p["_torch_softmax"]
    return (softmax * (grad - (grad * softmax).sum(-1, keepdim=True))).to(
        p["grad_output"].dtype
    )


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["grad_output"] = (
        torch.rand((a, b, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["m_init"] = (
        torch.rand((a, b, b), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.float32)
    p["num_iterations"] = c
    p["eps"] = 1e-6
    p["m_init"] = p["m_init"].softmax(-1) + p["eps"]
    p["_torch_state"] = p["m_init"].detach().requires_grad_()
    p["_torch_output"] = _torch_normalize(
        p["_torch_state"], p["num_iterations"], p["eps"]
    )
    p["_torch_incoming"] = p["grad_output"].float()
    p["_torch_softmax"] = p["m_init"] - p["eps"]
    return p


def _baseline(p):
    return _torch_sinkhorn_backward(p)


def _ftrain_sinkhorn_bwd(p):
    import flash_train.torch

    return flash_train.torch.sinkhorn_bwd(
        grad_output=p["grad_output"],
        m_init=p["m_init"],
        num_iterations=p["num_iterations"],
        eps=p["eps"],
    )


register(
    {
        "name": "sinkhorn_bwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (forward excluded)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.sinkhorn_bwd",
            "call": _ftrain_sinkhorn_bwd,
        },
        "suites": {
            "standard": [(4096, 4, 20), (8192, 4, 20), (16384, 4, 20), (32768, 4, 20)],
            "standard_v1": [
                (4096, 4, 20),
                (8192, 4, 20),
                (16384, 4, 20),
                (32768, 4, 20),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
