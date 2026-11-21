# flash-train 基准套件

为 README 算子橱窗、版本间性能曲线与性能回归门禁提供数据。基线一律为 **PyTorch 组合实现**（参考实现口径），不与其他厂商库对比。

## 口径分层

| 层 | 被测路径 | 受众 | 状态 |
|---|---|---|---|
| L1 Python 便利 | `flash_train.torch` 全流程（含每次调用组装开销） | Python 用户 | ✅ `run` |
| L2 C 便利 | `ftrainGemm` 全流程 | C 便利用户 | ✅ `cbench` |
| L3 Plan 复用 | 分阶段 C API，Plan 建好循环内执行推荐默认（primitive 0） | 分阶段 C 用户 | ✅ `cbench` |
| L4 逐 Primitive | Plan 内按下标枚举执行各实现 | Primitive / Finder 开发者 | ✅ `cbench` |

L1–L3 是同一组内核穿过不同 API 开销层的阶梯；L4 是选择中立的内核能力矩阵，同时是 Finder 偏好排序调优的数据源。四层共用同一标准形状网格（`cbench` 自动下发形状；`export-suites` 供手工运行 C harness 时复用同一网格）。

## 快速开始

前置：HCU 环境（DTK + PyTorch），已安装 `flash-train` wheel；图表生成可选装 matplotlib。
C harness 需先以 `-DFTRAIN_BUILD_BENCHMARKS=ON` 配置构建（产物 `build/bin/ftrain_bench_c`；未显式指定构建类型时仓库默认 Release——未优化构建的内核计时无意义）。
计时对并发负载敏感：跑基准前确认目标卡空闲，并用标准环境变量 `HIP_VISIBLE_DEVICES=N`
选定设备（Python 与 C 进程同时生效）。

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

- 结果 JSON（schema `ftrain-bench/1`）落在 `benchmarks/results/`，含完整环境元数据（卡型/arch、DTK、torch、wheel 版本、git SHA、时间戳）与形状集来源（套件名或 `custom`）；
- `git_sha` 记录被测二进制的构建提交；容器工作区通常不含 `.git`，跑基准时经 `FTRAIN_BENCH_GIT_SHA` 注入构建该 wheel 的提交；
- 结果随版本提交，是性能曲线与回归门禁的数据源；
- markdown 页面（`docs/benchmarks/*.md`）随库提交，图表 PNG 不入库（`.gitignore`）：合入 develop 后由 `bench-pages` workflow 从结果 JSON 重渲染并发布到 [GitHub Pages 站点](https://hygon-ai.github.io/flash-train/benchmarks/)，页面中的图即指向站点；
- 本地预览图表：`report`/`curve` 直接运行（PNG 生成在输出目录），传 `--pages-url ""` 可得到相对链接的页面；
- **引用任何数字必须带条件**（精度 · 形状 · 卡型），报告生成器会自动附加条件行。

## 限制

- L1 口径含每次调用的参数组装与计划创建开销：大形状下相对内核可忽略，小形状下偏保守——对外橱窗数字采用本口径（最接近"三行代码接入"用户的真实所得）；
- 迭代间存在缓存驻留，绝对值偏乐观于冷启动场景；
- matplotlib 缺失时仅输出表格。

## MoE 多卡基准

`ag_group_gemm` 和 `moe_reduce_rs` 使用相同的 `run`、`cbench`、`report`、`curve`
入口和 `ftrain-bench/1` JSON，但由单节点多进程执行。要求所有参与设备为 gfx936，
Python 扩展启用 MoE，Torch 具有 Gloo 和 NCCL/RCCL 后端。`run` 自动创建 Gloo 元数据组
和 NCCL/RCCL 参考通信组，不需要外部 `torchrun`。`cbench` 的原生 worker 不依赖 Torch，
由 Python 父进程采集环境信息和汇总；必须启用 `FTRAIN_ENABLE_MOE=ON` 构建 C harness。
两者都按调用者的可见设备顺序使用前 `--world-size` 张卡，不覆盖设备可见性配置。

MoE C worker 的 `FTRAIN_BENCH_ROUTE`、`FTRAIN_BENCH_SEED` 和 `FTRAIN_BENCH_RANK_M`
由 `cbench` 启动器设置，必须非空；缺失时会在初始化 GPU 前输出变量名并以状态码 2 退出，
不使用隐式默认值改变测试条件。

MoE 专用实现集中在以下目录，公共驱动只通过注册和通用会话钩子调用：

- `ftrain_bench/ops/moe/`：工作负载注册、多卡编排、Python worker 和结果标签；
- `csrc/ops/moe/`：C harness 算子会话、设备选择、固定采样策略和通信 bootstrap；
- `../tests/python/moe/benchmark_test.py`：基准编排、统计与报告回归测试。

`run`、`cbench`、`report`、`curve` 命令不变；报告仍兼容未记录结果标签的历史 JSON。

```bash
cmake -S . -B build/moe-native -DFTRAIN_BUILD_BENCHMARKS=ON
cmake --build build/moe-native --target ftrain_bench_c -j8
cd benchmarks
python3 -m ftrain_bench run --op ag_group_gemm --suite smoke \
    --route ag_nn_ring_chunk --experts 8 --topk 2 --warmup 2 --iters 5
python3 -m ftrain_bench cbench --op ag_group_gemm --suite smoke \
    --route ag_nn_ring_chunk --experts 8 --topk 2 --warmup 2 --iters 5 \
    --bin ../build/moe-native/bin/ftrain_bench_c --baseline 'results/ag_group_gemm-bf16-*.json'
python3 -m ftrain_bench run --op moe_reduce_rs --suite smoke \
    --route rs_nn --experts 8 --topk 2 --warmup 2 --iters 5
python3 -m ftrain_bench cbench --op moe_reduce_rs --suite smoke \
    --route rs_nn --experts 8 --topk 2 --warmup 2 --iters 5 \
    --bin ../build/moe-native/bin/ftrain_bench_c --baseline 'results/moe_reduce_rs-bf16-*.json'
python3 -m ftrain_bench report --in 'results/ag_group_gemm-bf16-*.json' --out ../docs/benchmarks
```

### MoE 输入与路线

- `--shapes` 始终为 **全局逻辑 M×K×N**，例如 `16384x16384x16384`；`standard`
  使用此形状，`32k` 使用 `32768x32768x32768`，`smoke` 为小形状接线验证。
  大形状不保证显存足够，尤其 RS 的 PyTorch 中间展开输出；可用 `--shapes` 选择实际目标，
  不因参考实现 OOM 改为遗漏步骤的基线。
- AG 在计时前按逻辑 topk 展开，每 rank 按 expert 排序：算子的 M 是逻辑 M×topk、
  K 是全局 K、N 是全局 N/TP、算子 topk=1；输出不再添加无关的还原步骤。
  PyTorch 基线保留合并后的 expert-major 输出；原生 ABI 输出为 rank-expert 顺序。
  仅在计时外对齐两者输出用于数值检查，不给 AG 基线额外增加输出反重排。
- RS 的算子 M 是逻辑 M、K 是全局 K/TP、N 是全局 N；输入是 token/choice 展开顺序。
  概率加权、token 归并和 ReduceScatter 都在计时内。
- `--route` 支持全部八条 AG 和两条 RS 现有显式路线，默认分别为
  `ag_nn_ring_chunk`、`rs_nn`；NN 原始权重为 `[E,K,N]`，NT 为 `[E,N,K]`，
  其中 K/N 按上面算子局部维度解释，不做权重预转置。
- 默认 `--world-size 8 --experts 64 --topk 6 --balanced-topk false --seed 2026`，
  unique_topk 固定 false。`--partition ragged` 生成 rank 间不均分 M；原始
  `ag_nt_a2a_chunk` 仍拒绝该模式，`_32k` 路线要求全局 K=32768。
- Python 和 C++ 使用同一个确定性 mixbits 数据规则。JSON 保留每个 rank 的四组
  字节抽样指纹（input/weight/routing/probabilities），不是完整内存的密码学 hash。
  C 结果只在完整 workload key、`FTRAIN_MOE_*` 调度环境和抽样指纹均匹配时配对 Python 基线。

### MoE 计时与结果

L1 是持久化 Python session 的 `forward`；L2 是持久化 C session 的 Execute；
L3 在计时前创建 Plan；L4 执行该 Plan 唯一显式路线的 Primitive。
**十条路线由十次显式 `--route` 运行覆盖**，不假定某个 Plan 会枚举所有路线。
session 初始化、输入生成、EP All-to-All 和销毁均不在稳态计时内。

MoE 使用 `--warmup 10 --iters 40` 固定次数，各 rank 完全一致，**不使用单卡自适应次数**。
预热/测量阶段边界做 CPU barrier，barrier 在事件计时外，不在算子内部增加同步，
不改原生通算重叠。每轮记录完成事件并等待完成。

- 主统计：每一轮取所有 rank 的事件耗时最大值，再计算 median、mean、min、p10、p90。
- 同时记录逐 rank 原始样本和 pooled 全 rank 统计；历史 pooled mean 只能与新的
  `ftrain.pooled.mean_ms` 同口径对比，不能与 rank-max median 混用。
- 基线是 PyTorch 组合全流程，不是 TE：AG 包括 AllGather、合并来源的 expert 数据块和
  expert matmul；RS 包括 expert matmul、输出顺序恢复、概率加权/归并及 ReduceScatter。
  ragged 分区的参考通信使用 padding，padding 开销计入基线；expert 索引元数据在计时前准备。
- 每次执行重做必要的数据搬运和重排，不把中间结果缓存后冒充全流程。
  采样前后检查输出有限性，并在计时外做数值比较；大形状仅抽样比较，完整正确性仍看 `tests/`。
- JSON 和报告分组 key 包括 route、TP、experts、topk、完整 rank M 分区、seed、布局和 chunks。
  每份 report/curve 只接受单一算子和精度，避免不同工作负载被错误合并。

逐 rank 原始 JSON、stderr 和配置保留在输出目录的 `*-logs/`。任一 rank 失败或超时后，
父进程只清理本次创建的进程组，运行失败，不生成成功结果；不结束外部 GPU 任务。
性能测量只在目标卡空闲时手动运行，不放进默认正确性 CTest。
GEMM 原来的单卡路径、形状集和自适应计时策略保持不变。

## 为新算子接入基准

两侧接入，逐项步骤见 [docs/development.md](../docs/development.md) 第 6 步：

- Python：`ftrain_bench/ops/` 新建模块，声明 `make_args`（固定种子生成张量）、`baseline`（PyTorch 组合实现）、`ftrain`（被测调用，被测库在调用点才导入）与 `suites`（标准形状集），并在 `ops/__init__.py` 导入注册；
- C harness：`benchmarks/csrc/ops/<op>.cpp` 实现 `OpCase`/`OpSession` 三个钩子（参照 `gemm.cpp`），并在 `benchmarks/CMakeLists.txt` 登记源文件；C 二进制用法为 `ftrain_bench_c <op> <precision> <m,k,n> [...]`（通常经 `cbench` 间接调用）。

形状集一经发布即固定，改动视为口径变更并需在结果元数据中可见。
