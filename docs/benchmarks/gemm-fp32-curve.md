# gemm 性能曲线（fp32）

> 最近环境：BW1102（unknown） · torch 2.7.1 · flash-train 0.1.0 · 2026-09-29；聚合全部历史结果（`benchmarks/results/`）自动生成。

> 当前仅一个版本（0.1.0+d9017e8c）的数据——曲线将随版本发布自动延长。

## L1 Python 便利层（端到端）

| 形状 (m×k×n) | 0.1.0+d9017e8c |
|---|---|
| 64x64x64 | 0.031 |
| 128x128x128 | 0.048 |
| 256x256x256 | 0.088 |

![L1 Python 便利层（端到端） 曲线](https://hygon-ai.github.io/flash-train/benchmarks/gemm-fp32-convenience-python-curve.png)

## L2 C 便利层（端到端）

| 形状 (m×k×n) | 0.1.0+d9017e8c |
|---|---|
| 64x64x64 | 0.027 |
| 128x128x128 | 0.046 |
| 256x256x256 | 0.085 |

![L2 C 便利层（端到端） 曲线](https://hygon-ai.github.io/flash-train/benchmarks/gemm-fp32-convenience-c-curve.png)

## L3 Plan 复用（稳态）

| 形状 (m×k×n) | 0.1.0+d9017e8c |
|---|---|
| 64x64x64 | 0.022 |
| 128x128x128 | 0.040 |
| 256x256x256 | 0.080 |

![L3 Plan 复用（稳态） 曲线](https://hygon-ai.github.io/flash-train/benchmarks/gemm-fp32-plan-reuse-curve.png)

## L4 逐 Primitive（实现矩阵）

| 形状 (m×k×n) · 实现 | 0.1.0+d9017e8c |
|---|---|
| 64x64x64 #0 Fp32Gemm | 0.022 |
| 128x128x128 #0 Fp32Gemm | 0.040 |
| 256x256x256 #0 Fp32Gemm | 0.080 |

![L4 逐 Primitive（实现矩阵） 曲线](https://hygon-ai.github.io/flash-train/benchmarks/gemm-fp32-primitive-curve.png)

数据来源：`benchmarks/results/`；口径与公平性规则见 [benchmarks/README.md](../../benchmarks/README.md)。
