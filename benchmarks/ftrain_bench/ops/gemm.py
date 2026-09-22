# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""GEMM 基准：基线 torch.matmul，被测 ftrain_torch.gemm（便利层端到端）。

形状元组语义为 (m, k, n)：a 为 m×k，b 为 k×n，结果为 m×n。
基线与被测均不带 beta 累加项（c=None），alpha 固定 1.0，保证严格对齐。
"""

import torch

import ftrain_torch

from . import register

DTYPES = {"fp32": torch.float32}

_SEED = 2026


def _make_args(shape, dtype):
    m, k, n = shape
    torch.manual_seed(_SEED)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    return a, b


register(
    {
        "name": "gemm",
        "precisions": ["fp32"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "torch.matmul", "call": lambda t: torch.matmul(t[0], t[1])},
        "ftrain": {"name": "ftrain_torch.gemm", "call": lambda t: ftrain_torch.gemm(t[0], t[1])},
        "suites": {
            "standard": [
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192),
                (1024, 4096, 4096),
                (4096, 1024, 11008),
            ]
        },
    }
)
