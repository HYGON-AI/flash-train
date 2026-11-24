# Flash Train

[![CI (HCU)](https://github.com/HYGON-AI/flash-train/actions/workflows/ci.yml/badge.svg)](https://github.com/HYGON-AI/flash-train/actions/workflows/ci.yml)

Flash Train（`flash-train`）是 HCU 上的新算子首发库：新 SOTA 模型带来的新算子，这里最先有高性能开源实现、参考实现与持续性能优化。

对用户的承诺分三层：

- **最先可用**——新算子发布后，在 HCU 上第一个可用的开源实现；
- **可信可用**——每个实现都附带 PyTorch 参考实现，数值正确性可独立验证；
- **越来越快**——算子落地后随版本持续调优，用户代码零改动获得性能提升。

## 为什么需要它

- 新架构模型发布后，关键新算子在 HCU 上长期缺少实现；
- 用 PyTorch 原生算子组合替代，训练吞吐差一个数量级；
- 自研 HIP 内核需要同时理解算法拓扑与硬件微架构，门槛高、周期以月计；
- 算子落地只是开始，还需要长期跟上新卡与新的优化机会。

## 算子目录

| 算子 | 来源 | 状态 | vs PyTorch 组合实现 | 验证/调优卡型 |
|---|---|---|---|---|
| sparse_attn_bwd（BF16） | DSA | 基础实现 | 1.5x | gfx936 |

HIP 实现可源码构建到全系列 HCU，本表只列出已完成正确性验证与性能调优的卡型。

## 性能

基准数据与跨版本性能曲线随版本提交到 `benchmarks/results/`，渲染后见 [docs/benchmarks/](docs/benchmarks/) 与[文档站](https://hygon-ai.github.io/flash-train/)：事件计时中位数，基线为 PyTorch 组合实现，方法论与复现步骤见 [benchmarks/README.md](benchmarks/README.md)。

`sparse_attn_bwd` 的基准结果尚未入库；首批结果提交后，数据与曲线会出现在上述位置。

## 快速上手

在 HCU 环境（DTK + AICC + HCU PyTorch）中构建并安装 wheel：

```bash
python3 -m pip install "scikit-build-core>=1.0" "cmake>=3.29" ninja pybind11
PYTHON=python3 bash build_wheel.sh
python3 -m pip install --no-deps --force-reinstall dist/flash_train-*.whl
```

编译入口为 `.github/scripts/_build_hcu_wheel.sh`，根目录 `build_wheel.sh` 为兼容入口。可用 `--check-only` 检查环境，`--out-dir DIR` 指定产物目录，`--build-tests` 同时编译 C++ 测试。CI 矩阵为 DTK 26.10 / Python 3.12 / Torch 2.11.0、2.13.0，定义在 `.github/workflows/ci.yml`：PR 和手动触发执行测试，develop 推送构建并发布 NIGHTLY。

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

两级接入（框架绑定、分阶段 C API）与环境变量开关详见 [docs/usage.md](docs/usage.md)。

## 架构一瞥

用户**调用某个算子的 API**，库负责**选哪个实现**（引擎 + 选择机制 + 选择缓存）并**异步执行**（Primitive + 内核）。两级接入最终执行同一条引擎路径：

- 每个算子有一套专属的句柄与函数：分阶段 C API 把 Plan 的创建和执行分开，Plan 建好后在训练热循环中复用；
- 新算子以“家族”为单位接入：Problem 校验 + Finder 选择策略 + 进程级 Engine + 平台实现；
- 平台相关代码全部收拢在平台子树，当前提供 HCU（HIP）平台。

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/development.md](docs/development.md)。

## 路线图

- **新算子首发**：持续跟进新 SOTA 模型的新算子，欢迎提 [issue](https://github.com/HYGON-AI/flash-train/issues) 告诉我们你最需要哪个；
- **性能**：已支持算子的持续优化，基准数据随版本发布；
- **生态**：PyTorch 绑定已提供，更多框架集成为长期方向；
- **硬件**：跟随 HCU 新卡型做适配与调优。

## 目录

- `include/flash_train`：公共 C 接口。
- `src`：算子库实现（含内部 C++ 接口与公共 C 接口实现）。
- `python`：框架绑定（当前 PyTorch）与 wheel 打包。
- `cmake`：构建与安装配置。
- `docs`：使用、开发与架构文档。
- `tests`：测试。

## 参与贡献

- 请求支持新算子或报告问题：提 [issue](https://github.com/HYGON-AI/flash-train/issues)。
- 接入新算子、新实现或新选择启发式：扩展路径见 [docs/development.md](docs/development.md)。

## 许可证

本项目以 [Apache License 2.0](LICENSE.txt) 发布，Copyright (c) 2026 Hygon Information Technology Co., Ltd.。各源文件头部带有对应的 SPDX 标识。

运行期第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
