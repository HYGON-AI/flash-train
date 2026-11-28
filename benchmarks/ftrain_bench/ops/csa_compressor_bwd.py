# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""csa_compressor_bwd 基准：基线为本文件的 Torch 组合，被测 flash_train.torch.csa_compressor_bwd（便利层端到端）。

参考来源：fused_dsa/test_csa_fused_compressor.py（test_dsa_kernels.py 未直接覆盖 compressor）
_eager_pool(mode="fp32") 的 autograd 梯度；cuDNN CSA / HIP compressor 对拍；上述 CUDA/HIP 后端未纳入当前计时基线。
前向与计算图在 make_args 中一次性构建，不计时；基线仅重复执行反向及输出类型转换。

形状元组 (tokens,ratio,input_width) 即 (T,R,coff*D)；KV / score 为 (T,coff*D)，输出为 (T//R,D)。本地预设保留 coff=1/2。
形状来源：DeepSeek-V4-Flash/Pro、GLM-5.2 官方配置，覆盖完整头数与 TP=8 的本地头数。
standard_v1 为模型形状集 v1（2026-10-14）；可比基准使用 --suite standard_v1，版本写入现有 meta.suite。
standard 保留为开发便捷入口；发布后 v1 的形状与本地预设一起固定，口径变化新增套件版本。
缩放与数据生成沿用本模块；本地预设为布局参数，公共流程不读取这些参数。
"""

import torch
from . import register

DTYPES = {"bf16": torch.bfloat16}


def _torch_compressor(kv, score, ape, ratio, head_dim=None, coff=1):
    groups = kv.shape[0] // ratio
    keys = kv[: groups * ratio].float().reshape(groups, ratio, -1)
    logits = (
        score[: groups * ratio].float().reshape(groups, ratio, -1) + ape.float()[None]
    )
    if coff == 2:
        d = head_dim
        previous_keys = torch.cat(
            (torch.zeros_like(keys[:1, :, :d]), keys[:-1, :, :d]), 0
        )
        previous_logits = torch.cat(
            (torch.full_like(logits[:1, :, :d], -torch.inf), logits[:-1, :, :d]), 0
        )
        keys = torch.cat((previous_keys, keys[:, :, d:]), 1)
        logits = torch.cat((previous_logits, logits[:, :, d:]), 1)
    return (keys * logits.softmax(1)).sum(1)


def _torch_compressor_backward(p):
    names = ("kv", "score", "ape")
    grads = torch.autograd.grad(
        p["_torch_output"], p["_torch_inputs"], p["_torch_incoming"], retain_graph=True
    )
    return tuple(t.to(p[n].dtype) for t, n in zip(grads, names))


# Model presets; custom triples retain the default layout.
_STANDARD_PARAMETERS = {
    (4096, 4, 256): {"coff": 2, "head_dim": 128},
    (4096, 4, 1024): {"coff": 2, "head_dim": 512},
    (4096, 128, 128): {"coff": 1, "head_dim": 128},
    (4096, 128, 512): {"coff": 1, "head_dim": 512},
    (8192, 4, 256): {"coff": 2, "head_dim": 128},
    (8192, 4, 1024): {"coff": 2, "head_dim": 512},
    (8192, 128, 128): {"coff": 1, "head_dim": 128},
    (8192, 128, 512): {"coff": 1, "head_dim": 512},
    (16384, 4, 256): {"coff": 2, "head_dim": 128},
    (16384, 4, 1024): {"coff": 2, "head_dim": 512},
    (16384, 128, 128): {"coff": 1, "head_dim": 128},
    (16384, 128, 512): {"coff": 1, "head_dim": 512},
}


def _make_args(shape, dtype):
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("shape expects three positive dimensions")
    a, b, c = shape
    config = _STANDARD_PARAMETERS.get(tuple(shape), {})
    coff = config.get("coff", 1)
    head_dim = config.get("head_dim", c)
    generator = torch.Generator(device="cuda").manual_seed(2026)
    p = {}
    p["kv"] = (torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.bfloat16
    )
    p["score"] = (
        torch.rand((a, c), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.bfloat16)
    p["ape"] = (torch.rand((b, c), device="cuda", generator=generator) * 0.2 - 0.1).to(
        torch.float32
    )
    p["cu_seqlens"] = torch.tensor([0, a], device="cuda", dtype=torch.int32)
    p["cu_seqlens_comp"] = torch.tensor([0, a // b], device="cuda", dtype=torch.int32)
    p["grad_compressed"] = (
        torch.rand((a // b, head_dim), device="cuda", generator=generator) * 0.2 - 0.1
    ).to(torch.bfloat16)
    p["ratio"] = b
    p["head_dim"] = head_dim
    p["coff"] = coff
    p["_torch_inputs"] = tuple(
        p[n].detach().float().requires_grad_() for n in ("kv", "score", "ape")
    )
    p["_torch_output"] = _torch_compressor(
        *p["_torch_inputs"], p["ratio"], p["head_dim"], p["coff"]
    )
    p["_torch_incoming"] = p["grad_compressed"].float()
    return p


def _baseline(p):
    return _torch_compressor_backward(p)


def _ftrain_csa_compressor_bwd(p):
    import flash_train.torch

    return flash_train.torch.csa_compressor_bwd(
        kv=p["kv"],
        score=p["score"],
        ape=p["ape"],
        cu_seqlens=p["cu_seqlens"],
        cu_seqlens_comp=p["cu_seqlens_comp"],
        grad_compressed=p["grad_compressed"],
        ratio=p["ratio"],
        head_dim=p["head_dim"],
        coff=p["coff"],
    )


register(
    {
        "name": "csa_compressor_bwd",
        "precisions": ["bf16"],
        "dtypes": DTYPES,
        "make_args": _make_args,
        "baseline": {"name": "PyTorch backward (forward excluded)", "call": _baseline},
        "ftrain": {
            "name": "flash_train.torch.csa_compressor_bwd",
            "call": _ftrain_csa_compressor_bwd,
        },
        "suites": {
            "standard": [
                (4096, 4, 256),
                (4096, 4, 1024),
                (4096, 128, 128),
                (4096, 128, 512),
                (8192, 4, 256),
                (8192, 4, 1024),
                (8192, 128, 128),
                (8192, 128, 512),
                (16384, 4, 256),
                (16384, 4, 1024),
                (16384, 128, 128),
                (16384, 128, 512),
            ],
            "standard_v1": [
                (4096, 4, 256),
                (4096, 4, 1024),
                (4096, 128, 128),
                (4096, 128, 512),
                (8192, 4, 256),
                (8192, 4, 1024),
                (8192, 128, 128),
                (8192, 128, 512),
                (16384, 4, 256),
                (16384, 4, 1024),
                (16384, 128, 128),
                (16384, 128, 512),
            ],
        },
        "tolerances": {"bf16": (0.05, 0.005)},
    }
)
