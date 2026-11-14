# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""KDA 基准：基线为 PyTorch 组合实现（flash_train.torch.kda.kda_chunked_reference，fp32 chunk 形式），
被测为 flash_train.torch.kimi_delta_attention；两者都跑前向 + 反向（训练口径）。

形状元组语义为 (T, H)：单条序列，K = V = 128，Kimi K3 门控（A_log/dt_bias，lower_bound=-5）。
"""

import torch

from . import register

DTYPES = {"bf16": torch.bfloat16}

_SEED = 2026


def _make_args(shape, dtype):
    T, H = shape
    gen = torch.Generator(device="cuda").manual_seed(_SEED)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k, v = ((r(T, H, 128) * 0.3).to(dtype) for _ in range(3))
    g = (r(T, H, 128) * 0.3).to(dtype)
    beta = r(T, H).sigmoid()
    a_log = torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen))
    dt_bias = r(H * 128) * 0.1
    d_o = r(T, H, 128).to(dtype)
    return q, k, v, g, beta, a_log, dt_bias, d_o


def _fwd_bwd(fn, t):
    leaves = [x.detach().requires_grad_(True) for x in t[:7]]
    q, k, v, g, beta, a_log, dt_bias = leaves
    o = fn(q, k, v, g, beta, A_log=a_log, dt_bias=dt_bias)
    torch.autograd.grad(o, leaves, t[7].to(o.dtype))


# 被测库在调用点才导入：cbench/report/curve 等编排路径不依赖 wheel 已安装
def _ftrain_kda(t):
    import flash_train.torch

    _fwd_bwd(flash_train.torch.kimi_delta_attention, t)


def _torch_kda(t):
    from flash_train.torch.kda import kda_chunked_reference

    _fwd_bwd(kda_chunked_reference, t)


register(
    {
        "name": "kda",
        "precisions": ["bf16"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch chunked KDA (fwd+bwd)", "call": _torch_kda},
        "ftrain": {"name": "flash_train.torch.kimi_delta_attention (fwd+bwd)", "call": _ftrain_kda},
        "suites": {"standard": [(4096, 12), (8192, 12), (8192, 24), (8192, 48), (16384, 24), (32768, 12)]},
    }
)
