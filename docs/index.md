# Flash Train

Flash Train（`flash-train`）是 HCU 上的新算子首发库：新 SOTA 模型带来的新算子，这里最先有高性能开源实现、参考实现与持续性能优化。

- **最先可用**——新算子发布后，在 HCU 上第一个可用的开源实现；
- **可信可用**——每个实现都附带 PyTorch 参考实现，数值正确性可独立验证；
- **越来越快**——算子落地后随版本持续调优，用户代码零改动获得性能提升。

## 算子目录

| 算子 | 来源 | 状态 | vs PyTorch 组合实现 | 验证/调优卡型 |
|---|---|---|---|---|
| sparse_attn_bwd（BF16） | DSA | 基础实现 | 1.5x | gfx936 |

HIP 实现可源码构建到全系列 HCU，本表只列出已完成正确性验证与性能调优的卡型。

## 快速上手

在 HCU 环境（DTK + PyTorch）中构建并安装 wheel：

```bash
python3 -m pip wheel . --no-deps -w dist
python3 -m pip install --no-deps dist/flash_train-*.whl
```

计算 sparse_attn_bwd（`out`、`lse` 是前向输出；这里用 Torch 组合前向把它们准备出来）：

```python
import torch, flash_train.torch

T, H, S, D, K = 8, 8, 8, 512, 17
q = (torch.randn(T, H, D, device="cuda") * 0.1).to(torch.bfloat16)
kv = (torch.randn(S, D, device="cuda") * 0.1).to(torch.bfloat16)
sink = torch.zeros(H, device="cuda", dtype=torch.float32)
indices = (torch.arange(K, device="cuda", dtype=torch.int32) % S).expand(T, -1).contiguous()
scale = D ** -0.5
keys = kv.float()[indices.long()]
scores = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
lse = scores.logsumexp(-1)                    # 不含 sink 的 logsumexp
logits = torch.cat([scores, sink[None, :, None].expand(T, -1, -1)], -1)
out = torch.einsum("thk,tkd->thd", logits.softmax(-1)[..., :-1], keys).to(torch.bfloat16)
dout = torch.randn_like(out)

dq, dkv, dsink = flash_train.torch.sparse_attn_bwd(
    q, kv, out, dout, lse, sink, indices, softmax_scale=scale
)
```

## 继续阅读

- [使用指南](usage.md)——三级接入（框架绑定、便利 C API、分阶段 C API）与环境变量；
- [架构](architecture.md)——引擎 / Primitive 分层与选择机制；
- [开发指南](development.md)——新算子接入路径与扩展点；
- [基准套件](https://github.com/HYGON-AI/flash-train/blob/develop/benchmarks/README.md)——口径、复现步骤与结果数据。

## 反馈与参与

- 请求支持新算子或报告问题：提 [issue](https://github.com/HYGON-AI/flash-train/issues)；
- 本项目以 [Apache License 2.0](https://github.com/HYGON-AI/flash-train/blob/develop/LICENSE.txt) 发布。
