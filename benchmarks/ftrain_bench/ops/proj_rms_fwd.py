# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""proj_rms_fwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.proj_rms_fwd（便利层端到端）。

参考来源：Megatron-LM/tests/unit_tests/fusions/test_fused_mhc_kernels.py
_ref_proj_rms；native / Triton / cuTile 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。

形状元组 (rows,outputs,reduction) 即 (M,N,K)；x 为 (M,K)，weight 为 (N,K)，投影为 (M,N)。
形状来源：fused_mHC/tests/test_performance.py 与 tests/benchmark.py 的 large 配置；后者 K=n*C。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
eps=1e-6；r 为 1/(RMS(x)+eps)。
数据和精度沿用本模块；这里只转换形状，公共计时流程不变。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _torch_projection(x, weight, eps):
    proj = x.float() @ weight.float().T
    r = 1 / (x.float().square().mean(-1, keepdim=True).sqrt() + eps)
    return proj, r


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
        torch.rand((b, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(dtype)
    p["eps"] = 1e-6
    return p


def _baseline(p):
    return tuple(
        t.to(p["x"].dtype) for t in _torch_projection(p["x"], p["weight"], p["eps"])
    )


def _ftrain_proj_rms_fwd(p):
    import flash_train.torch

    return flash_train.torch.proj_rms_fwd(x=p["x"], weight=p["weight"], eps=p["eps"])


register(
    {
        "name": "proj_rms_fwd",
        "precisions": ["bf16", "fp16", "fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch composition", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.proj_rms_fwd",
            "call": _ftrain_proj_rms_fwd,
        },
        "suites": {
            "standard": [
                (4096, 24, 16384),
                (8192, 24, 16384),
                (16384, 24, 16384),
                (32768, 24, 16384),
                (4096, 24, 28672),
                (8192, 24, 28672),
                (16384, 24, 28672),
                (32768, 24, 28672),
            ],
            "standard_v1": [
                (4096, 24, 16384),
                (8192, 24, 16384),
                (16384, 24, 16384),
                (32768, 24, 16384),
                (4096, 24, 28672),
                (8192, 24, 28672),
                (16384, 24, 28672),
                (32768, 24, 28672),
            ],
        },
        "tolerances": {
            "fp32": (0.0002, 2e-05),
            "fp16": (0.02, 0.002),
            "bf16": (0.05, 0.005),
        },
    }
)
