# Flash Train

Flash Train（`flash-train`）是 HCU 上的新算子首发库：新 SOTA 模型带来的新算子，这里最先有高性能开源实现、参考实现与持续性能优化。

- **最先可用**——新算子发布后，在 HCU 上第一个可用的开源实现；
- **可信可用**——每个实现都附带 PyTorch 参考实现做数值对拍，正确性可独立验证；
- **越来越快**——算子落地后随版本持续调优，用户代码零改动获得性能提升。

## 算子目录

| 算子 | 来源 | 状态 | vs PyTorch 组合实现 | 验证/调优卡型 |
|---|---|---|---|---|
| GEMM（FP32） | — | 链路验证样例 | — | gfx938 |

HIP 实现可源码构建到全系列 HCU，本表只列出已完成正确性验证与性能调优的卡型。

## 快速上手

在 HCU 环境（DTK + PyTorch）中构建并安装 wheel：

```bash
python3 -m pip wheel . --no-deps -w dist
python3 -m pip install --no-deps --force-reinstall dist/flash_train-*.whl
```

计算 GEMM：

```python
import torch, flash_train.torch

a = torch.randn(128, 64, device="cuda")   # m×k
b = torch.randn(64, 96,  device="cuda")   # k×n

d = flash_train.torch.gemm(a, b)   # d = a@b；支持 alpha/beta 与可选累加项
```

## 继续阅读

- [性能基准](benchmarks/gemm-fp32.md)——事件计时中位数、PyTorch 组合基线、[跨版本曲线](benchmarks/gemm-fp32-curve.md)；
- [使用指南](usage.md)——三级接入（框架绑定、便利 C API、分阶段 C API）与环境变量；
- [架构](architecture.md)——Pattern/引擎/Primitive 分层与选择机制；
- [开发指南](development.md)——新算子接入路径与扩展点。

## 反馈与参与

- 请求支持新算子或报告问题：提 [issue](https://github.com/HYGON-AI/flash-train/issues)；
- 本项目以 [MIT License](https://github.com/HYGON-AI/flash-train/blob/develop/LICENSE) 发布。
