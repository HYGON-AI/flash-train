# Flash Train

[![Static Checks](https://github.com/HYGON-AI/flash-train/actions/workflows/static-checks.yml/badge.svg)](https://github.com/HYGON-AI/flash-train/actions/workflows/static-checks.yml)

Flash Train（`flash-train`）是 HCU 上的新算子首发库：新 SOTA 模型带来的新算子，这里最先有高性能开源实现、参考实现与持续性能优化。

对用户的承诺分三层：

- **最先可用**——新算子发布后，在 HCU 上第一个可用的开源实现；
- **可信可用**——每个实现都附带 PyTorch 参考实现做数值对拍，正确性可独立验证；
- **越来越快**——算子落地后随版本持续调优，用户代码零改动获得性能提升。

## 为什么需要它

- 新架构模型发布后，关键新算子在 HCU 上长期缺少实现；
- 用 PyTorch 原生算子组合替代，训练吞吐差一个数量级；
- 自研 HIP 内核需要同时理解算法拓扑与硬件微架构，门槛高、周期以月计；
- 算子落地只是开始，还需要长期跟上新卡与新的优化机会。

## 算子目录

| 算子 | 来源 | 状态 | vs PyTorch 组合实现 | 验证/调优卡型 |
|---|---|---|---|---|
| GEMM（FP32） | — | 链路验证样例 | — | gfx938 |

HIP 实现可源码构建到全系列 HCU，本表只列出已完成正确性验证与性能调优的卡型。

GEMM（FP32）当前用作接入层、选择引擎、缓存与 Plan 复用的全链路验证样例，正式算子落地后将让出示例位置并逐步退出。

## 快速上手

在 HCU 环境（DTK + PyTorch）中构建并安装 wheel：

```bash
python3 -m pip wheel . --no-deps -w dist
python3 -m pip install --no-deps --force-reinstall dist/ftrain_torch-*.whl
```

计算 GEMM：

```python
import torch, ftrain_torch

a = torch.randn(128, 64, device="cuda")   # m×k
b = torch.randn(64, 96,  device="cuda")   # k×n

d = ftrain_torch.gemm(a, b)               # d = a@b；支持 alpha/beta 与可选累加项
```

三级接入（PyTorch 绑定、便利 C API、分阶段 C API）与环境变量开关详见 [docs/usage.md](docs/usage.md)。

## 架构一瞥

用户描述**要算什么**（Pattern/Args），库负责**选哪个实现**（引擎 + 选择机制 + 选择缓存）并**异步执行**（Primitive + 内核）。三级接入最终执行同一条引擎路径：

- 分阶段 C API 把一次性的拓扑描述与每次调用的数据分开，Plan 建好后在训练热循环中复用；
- 新算子以“家族”为单位接入：Problem 校验 + Finder 选择策略 + 平台实现；
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
- `python`：PyTorch 绑定与 wheel 打包。
- `cmake`：构建与安装配置。
- `docs`：使用、开发与架构文档。
- `tests`：测试。

## 参与贡献

- 请求支持新算子或报告问题：提 [issue](https://github.com/HYGON-AI/flash-train/issues)。
- 接入新算子、新实现或新选择启发式：扩展路径见 [docs/development.md](docs/development.md)。

## 许可证

本项目以 [MIT License](LICENSE) 发布，Copyright (c) 2026 Hygon Information Technology Co., Ltd.。各源文件头部带有对应的 SPDX 标识。

运行期第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
