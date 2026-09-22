# flash-train 基准套件

为 README 算子橱窗、版本间性能曲线与性能回归门禁提供数据。基线一律为 **PyTorch 组合实现**（参考实现口径），不与其他厂商库对比。

## 口径分层

| 层 | 被测路径 | 受众 | 状态 |
|---|---|---|---|
| L1 Python 便利 | `ftrain_torch` 全流程（含每次调用组装开销） | Python 用户 | ✅ `run` |
| L2 C 便利 | `ftrainGemm` 全流程 | C 便利用户 | ✅ `cbench` |
| L3 Plan 复用 | 分阶段 C API，Plan 建好循环内执行推荐默认（primitive 0） | 分阶段 C 用户 | ✅ `cbench` |
| L4 逐 Primitive | Plan 内按下标枚举执行各实现 | Primitive / Finder 开发者 | ✅ `cbench` |

L1–L3 是同一组内核穿过不同 API 开销层的阶梯；L4 是选择中立的内核能力矩阵，同时是 Finder 偏好排序调优的数据源。四层共用同一标准形状网格（`export-suites` 导出给 C harness）。

## 快速开始

前置：HCU 环境（DTK + PyTorch），已安装 `ftrain_torch` wheel；图表生成可选装 matplotlib。
C harness 需先以 `-DFTRAIN_BUILD_BENCHMARKS=ON` 配置构建（产物 `build/bin/ftrain_bench_c`）。

```bash
cd benchmarks
python3 -m ftrain_bench run --op gemm --suite standard        # L1：落盘 results/*.json
python3 -m ftrain_bench cbench --op gemm --suite standard \
    --bin ../build/bin/ftrain_bench_c \
    --baseline results/gemm-fp32-*.json                      # L2/L3/L4：与 L1 同会话配对
python3 -m ftrain_bench report --in 'results/gemm-*.json' --out ../docs/benchmarks
python3 -m ftrain_bench curve  --in 'results/gemm-*.json' --out ../docs/benchmarks
python3 -m ftrain_bench run --op gemm --shapes 1024x4096x4096 # 自定义形状（m×k×n）
```

L2/L3 的基线取自 `--baseline` 指定的 L1 结果（同节点同会话跑两次配对）；L4 为实现矩阵，
不设外部基线。`curve` 聚合 `results/` 的全部历史结果生成跨版本曲线，随 release 数据积累
自动延长。大形状的耗时受被测内核性能支配（每档 预热 10 + 至多 200 次迭代），可用
`--shapes` 先跑小样本。

## 计时口径

- torch 当前流上事件计时，record→record→synchronize；
- 预热 10 次；迭代数自适应（目标累计 200ms，上下限 10~200 次）；
- **主口径 = 中位数**，同时记录 mean / min / p10 / p90 与实际迭代数；
- 基线与被测共用同一批张量（固定种子生成、循环内复用），测 steady state；
- 未做 L2 flush——双方同等条件，公平性优先于绝对值，比较时只看相对值。

## 结果与报告

- 结果 JSON（schema `ftrain-bench/1`）落在 `benchmarks/results/`，含完整环境元数据（卡型/arch、DTK、torch、wheel 版本、git SHA、时间戳）；
- 结果随版本提交，是性能曲线与回归门禁的数据源；
- **引用任何数字必须带条件**（精度 · 形状 · 卡型），报告生成器会自动附加条件行。

## 限制

- L1 口径含每次调用的参数组装与计划创建开销：大形状下相对内核可忽略，小形状下偏保守——对外橱窗数字采用本口径（最接近"三行代码接入"用户的真实所得）；
- 迭代间存在缓存驻留，绝对值偏乐观于冷启动场景；
- matplotlib 缺失时仅输出表格。

## 为新算子接入基准

在 `ftrain_bench/ops/` 新建模块：声明 `make_args`（固定种子生成张量）、`baseline`（PyTorch 组合实现）、`ftrain`（被测调用）与 `suites`（标准形状集），并在 `ops/__init__.py` 导入注册。形状集一经发布即固定，改动视为口径变更并需在结果元数据中可见。
