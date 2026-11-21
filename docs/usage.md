# flash-train 使用指南

本库面向三类使用者，按接入成本从低到高排列。三类入口最终执行同一套引擎选择与内核调度，只是封装层次不同。

| 你是谁 | 用哪套接口 | 一次调用的成本 | 适合场景 |
|---|---|---|---|
| PyTorch 用户 | `flash_train.torch`（wheel） | 每次调用含参数组装与计划缓存查询 | 快速接入、原型验证 |
| C/C++ 调用方（图省事） | `flash_train.h` 便利 API | 同上 | 偶发调用、不建 Plan 的场景 |
| C/C++ 调用方（追求性能） | `common.h` 分阶段 API | Plan 复用后仅剩执行 | 训练热循环 |

---

## 1. PyTorch 用户

### 安装

在 HCU 环境（DTK + PyTorch）中从源码构建 wheel 并安装：

```bash
pip wheel . --no-deps -w dist
pip install --no-deps dist/flash_train-*.whl
```

### 计算 GEMM

```python
import torch, flash_train.torch

a = torch.randn(128, 64,  device="cuda")   # m×k
b = torch.randn(64, 96,   device="cuda")   # k×n
c = torch.randn(128, 96,  device="cuda")   # m×n，可选

d = flash_train.torch.gemm(a, b, c, alpha=1.5, beta=0.25)   # d = α·(a@b) + β·c
d = flash_train.torch.gemm(a, b)                           # c=None 时 β·c 项被跳过
```

要点：

- 参数约束：`a/b/c` 为 **FP32、二维、连续、设备**张量，`a.size(1) == b.size(0)`，`c` 形状为 `m×n`；不满足抛 `RuntimeError`。
- 输出 `d` 由库分配并返回；计算提交在 **torch 当前流** 上，与相邻算子自然保序。
- `alpha`/`beta` 是 Python 标量，按 FP32 传入内核。
- 热循环请改用分阶段 API 复用 Plan（见第 3 节）；本接口每次调用有微秒级的参数组装开销。

### 计算 KDA（Kimi Delta Attention）

```python
import torch, flash_train.torch as ft

T, H = 8192, 12                                         # 打包后的总 token 数、head 数
q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
g = torch.randn_like(q)                                 # Kimi K3：原始 gate
beta = torch.rand(T, H, device="cuda")                  # 已过 sigmoid 的 beta
A_log = torch.zeros(H, device="cuda"); dt_bias = torch.zeros(H * 128, device="cuda")
cu = torch.tensor([0, 3000, 8192], dtype=torch.int32, device="cuda")   # 两条序列

o = ft.kimi_delta_attention(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu)
o.sum().backward()                                      # 支持 autograd：q/k/v/g/beta/A_log/dt_bias 的梯度
```

要点：

- 布局为打包 THD：`q/k/v/o` 是 `[T, H, 128]` 的 BF16；`cu_seqlens`（int32 `[N+1]`）把 T 行切成互不相关的序列，每条序列从零状态开始，长度任意（不必是 64 的倍数，可以为 0）。省略时视为一条序列。
- 两种 gate 契约：
  - 传 `A_log`/`dt_bias` 时，g 是原始 gate，算子内部计算 `lower_bound·sigmoid(exp(A_log)·(g+dt_bias))`，用于 Kimi K3，`lower_bound` 取 [-5, 0)；
  - 不传时，g 是 FP32 的逐 token log 衰减（cuDNN-frontend PR #556 契约），要求 ≥ -5；传 `check_decay=True` 可以检查这一点（默认关闭，开启后每次调用多一次设备归约和一次主机读取）。
- `use_qk_l2norm_in_kernel=True`（默认）表示在算子内部对 q/k 做 l2 归一化，eps 为 1e-6。
- 当前实现为 gfx936 手写内核（`G2KdaForward`/`G2KdaBackward`），只支持 K = V = 128，其他配置抛 `RuntimeError`（`FTRAIN_STATUS_UNSUPPORTED`）。确定性：只用固定顺序归约，不用原子操作。
- 精度：与 FP64 参考（`ft.kda_reference`）相比，相对 RMS 误差约 0.3–0.7%，判据为 o < 2e-2、dq/dk/dv < 4e-2、门控侧梯度 < 6e-2（与 PR #556 一致）。
- 暂不支持初始状态和最终状态（训练从零状态开始，K3 不使用）。

#### 前向为反向保存什么（`disable_recompute`，所有选项逐位一致）

默认由算子自己选择最优策略，**调用方不需要设置任何东西**。只有想在显存和速度之间做取舍时，才传 `disable_recompute`；参数含义与 FLA 相同（"反向是否不重算"），`kimi_delta_attention` 和 `chunk_kda` 都接受：

| `disable_recompute` | 前向留给反向的内容 | 新增显存（每 token·head，含 o） | 反向 |
|---|---|---:|---|
| 不传（`None`，默认） | H ≤ 64 时保存 h；H > 64 时不保存，但在可重入 checkpoint 的重算前向里交出完整状态（只活到这一层的反向结束） | H ≤ 64 约 1544 B；H > 64 约 1032 B | 速度最优（gfx936 实测） |
| `True` | 总是保存 h | 约 1544 B | 不跑递推 |
| `False` | 最小集合（与 FLA 默认的显存相同） | 约 1032 B | 重跑一次递推 |

- 统计口径：不含调用方本来就持有的 q/k/v/g/beta；qn/kn 按新分配计入（512 B）。模型里 q/k 如果只被 KDA 使用，保存 qn/kn 后 q/k 会被释放，这一项实际不增加显存。
- 在 no_grad 下（推理、checkpoint 的第一次前向），前向什么都不保存，与参数无关。
- 输出和所有梯度在三种选项下逐位相同。

```python
o, _ = chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                 use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
                 disable_recompute=False)          # 例如显存紧张时：保存量与 FLA 默认相同
```

### 在 Megatron 中使用 KDA（与 FLA 的 `chunk_kda` 对齐）

`flash_train.torch.chunk_kda` 的签名与 FLA 的 `fla.ops.kda.chunk_kda` 相同，输入为 `[B, T, H, 128]`，返回 `(o, None)`。Megatron 侧把 import 换成 flash-train 即可，调用参数不变，不需要设置任何环境变量：

```python
from flash_train.torch import chunk_kda          # 原来：from fla.ops.kda import chunk_kda
```

要求 gfx936 卡与 K = V = 128；其他架构抛 `RuntimeError`（`FTRAIN_STATUS_UNSUPPORTED`），不会回退到 FLA。

性能（BW1000，与同卡 FLA 0.5.2 比较，B = 1、K = V = 128，T = 1K–32K × H = 64/32/16/8 共 24 个形状）：前向 1.63–2.27 倍，反向 1.83–2.44 倍，前向 + 反向约 2.1 倍。

#### 输入形状：B / T / H 的含义

`[B, T, H, 128]` 沿用 FLA `chunk_kda` 的约定：B 是张量的 batch 维，T 是每条样本的序列长度，H 是本卡上的 KDA head 数（总 head 数 / TP）。q / k / v / g 都是 `[B, T, H, 128]`，beta 是 `[B, T, H]`，A_log 是 `[H]`，dt_bias 是 `[H*128]`（或 `[H, 128]`）。

- **定长（不传 `cu_seqlens`）**：B = micro batch size，T = seq_len，按 B 条长度为 T 的序列处理。
- **变长（packed / THD，传 `cu_seqlens`）**：B 必须为 1，T 是拼接后的总 token 数，`cu_seqlens`（int32 `[N+1]`）标出序列边界，各序列长度任意，不必是 64 的倍数。`cu_seqlens = [0, T]` 自动走更快的 dense kernel。

#### 与 Megatron 的配合

- **TP**：head 按 TP 切分，flash-train 不感知 TP，按传入的 H 计算。
- **全层重算**（`--recompute-granularity full`）：可重入 checkpoint 的第一次前向在 no_grad 下运行，不保存任何东西；重算前向按上一节的表保存，只活到这一层的反向结束。
- **确定性**：固定顺序归约、不用原子操作，同样的输入多次运行结果逐位相同。

#### 参数对照（与 FLA 的差别）

- 只做布局转换，与在打包 THD 张量上直接调用 `kimi_delta_attention` 逐位一致：`[B, T, H, D]` 展平成 `[B*T, H, D]`；传 `cu_seqlens` 时要求 `B == 1`（与 FLA 相同）。
- gate：`use_gate_in_kernel=True` 时要求给出 `A_log`、`dt_bias` 和 `lower_bound`（FLA 的 safe_gate 形式）；`use_gate_in_kernel=False` 时 g 是逐 token log 衰减。`use_beta_sigmoid_in_kernel=True` 时在宿主侧对 beta 做 sigmoid（`allow_neg_eigval` 时乘 2）。
- `disable_recompute` 与 FLA 含义相同，见上一节；不传时用算子自选的策略（FLA 的默认值是 `False`，这里是 `None`）。`chunk_size`、`cu_seqlens_cpu` 是 FLA 的调优参数，这里接受但不起作用。
- 不支持的参数直接抛异常，不会悄悄回退：`initial_state`、`output_final_state=True`、`return_intermediate_states`、不为 None 的 `cp_context`（即 CP > 1）、没有 `lower_bound` 的 softplus gate 抛 `NotImplementedError`；未知关键字抛 `TypeError`。


### KDA 调试开关（flash-train 开发用，训练不需要）

这些变量登记在 `include/flash_train/env.h`，只用于 A/B 实验和调试，不改变任何结果的数值。训练脚本里不要设置。

| 变量 | 默认 | 作用 |
|---|---|---|
| `FTRAIN_KDA_SAVE_QKN` | `1` | `0`：前向保存 q/k 而不是 qn/kn，反向重建 qn/kn |
| `FTRAIN_KDA_SAVE_H` | `auto` | `1` 总是保存 h，`0` 从不保存，`auto` 在 H ≤ 64 时保存 |
| `FTRAIN_KDA_KEEP_STATE` | `auto` | 不保存 h 时，`1` 总是交出完整状态，`0` 从不交出，`auto` 只在可重入 checkpoint 的重算前向里交出 |
| `FTRAIN_KDA_DEV` | 不设置 | 设为非空且不为 `0` 时，G2 kernel 才读取分阶段计时开关（`G2_*_STOP`，用于定位各阶段耗时）；不设置时这些开关一律取默认值 |

- 前三个只在调用方没有传 `disable_recompute` 时生效，每次前向调用时读取；`FTRAIN_KDA_DEV` 每个进程只在第一次使用时读取一次。

```bash
# 例：A/B 比较"保存 h"与"反向重算"的速度
FTRAIN_KDA_SAVE_H=1 python3 bench.py
FTRAIN_KDA_SAVE_H=0 python3 bench.py
# 例：只跑 prep 的前两个阶段，给单阶段计时
FTRAIN_KDA_DEV=1 G2_PREP_STOP=2 python3 bench.py
```

### 用 C API 接入 KDA（与 PyTorch 入口同一套端口）

`flash_train.h` 的 `ftrainKdaForward` / `ftrainKdaBackward`（以及对应的 `...GetWorkspaceBytes`）接收 `FTrainKdaForwardTensors` / `FTrainKdaBackwardTensors`；`common.h` 的 `ftrainPatternAddKdaForward` / `ftrainPatternAddKdaBackward` 用同样的端口顺序建 Pattern，用于复用 Plan。

| 端口 | 前向 | 反向 | 说明 |
|---|---|---|---|
| q, k, v, g, beta, a_log, dt_bias, cu_seqlens | 输入 | 输入 | 不用的端口传空内存（a_log/dt_bias 不在算子内算 gate 时；cu_seqlens 只有一条序列时） |
| o / d_o | 输出 | 输入 | `[T, H, 128]` bf16 |
| q_rstd, k_rstd, aqk, akk | 输出 | 输入 | 必须保存给反向 |
| qn, kn | 可选输出 | 可选输入 | 只在 l2norm_qk 时；给出后反向的 q/k 可以传空内存 |
| h | 可选输出 | 可选输入 | `ftrainKdaSavedHElements(T, H, N, has_cu)` 个 bf16，布局不公开 |
| state | 可选输出 | 可选输入 | `ftrainKdaStateElements(T, H, N, has_cu)` 个 bf16，布局不公开；与 h 互斥 |
| dq, dk, dv, dg, dbeta, da_log, ddt_bias | — | 输出 | dq/dk/dg 在调用期间兼作临时空间，不得与任何输入重叠 |

可选端口都是“前向写了、反向就用；传空内存就重算”，结果逐位一致。执行是异步的：返回后，所有张量和 workspace 必须保持有效，直到流上的计算完成。

---

## 2. 便利 C API（flash_train.h）

一个函数覆盖"拼参数→选实现→执行"全过程，等价于第 3 节的分阶段流程写一遍样板：

```c
#include <flash_train/flash_train.h>

FTrainStatus status = ftrainGemm(a, b, c, d, alpha, beta,
                                 FTRAIN_NUMERIC_TYPE_FP32,
                                 workspace, workspace_bytes, stream);
```

| 参数 | 要求 |
|---|---|
| `a`/`b`/`c`/`d` | 秩 2 设备 Tensor 视图（`FTrainStorageView`）；`c` 内存可为 `NULL`（跳过 `β·c`） |
| `alpha`/`beta` | 宿主机标量 Tensor（秩 0） |
| `compute_type` | 计算精度 |
| `workspace`/`workspace_bytes` | 推荐实现所需 workspace（当前 FP32 内核为 0，传 `NULL, 0` 即可） |
| `stream` | 执行流；函数不同步，返回后设备内存须保持有效直至该流完成 |

状态码语义与分阶段 API 完全一致：`FTRAIN_STATUS_INVALID_ARGUMENT`（参数不一致/workspace 不足）、`FTRAIN_STATUS_UNSUPPORTED`（无实现满足参数）、`FTRAIN_STATUS_SUCCESS`。失败详情可用 `ftrainGetLastStatus()` 查询。

---

## 3. 通用 C API（common.h）

分阶段 API 把**一次性的描述**与**每次调用的数据**分开，Plan 建好后在循环外复用：

```c
/* ---- 一次性：拓扑描述与匹配 ---- */
FTrainPattern pattern;  ftrainPatternCreate(&pattern);
FTrainTensorId a,b,c,d,alpha,beta;  FTrainGemmOpId gemm;
ftrainPatternAddTensor(pattern, &a);      /* ×6，顺序任意 */
/* ... */
ftrainPatternAddGemm(pattern, &gemm, a, b, c, d, alpha, beta);

FTrainOps ops;  ftrainOpsCreate(&ops, pattern);
ftrainPatternDestroy(pattern);            /* Ops 不依赖 Pattern 存活 */

/* ---- 每个形状一次：参数与计划 ---- */
FTrainArgs args;  ftrainArgsCreate(&args, ops);
ftrainArgsSetTensor(args, a, view_a);     /* ×6 */
ftrainArgsSetGemm(args, gemm, FTRAIN_NUMERIC_TYPE_FP32);

FTrainPlan plan;  ftrainPlanCreate(&plan, args);

/* ---- 每次迭代：按预算挑一个并执行 ---- */
uint64_t n, bytes;
ftrainPlanGetNumPrimitives(plan, &n);
for (uint64_t i = 0; i < n; ++i) {
    const char* name;  ftrainPlanGetPrimitiveName(plan, i, &name);  /* 甄别/日志用 */
    ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, i, &bytes);
    if (bytes <= my_budget) { ftrainPlanExecutePrimitive(plan, i, ws, my_budget, stream); break; }
}

ftrainPlanDestroy(plan);  ftrainArgsDestroy(args);  ftrainOpsDestroy(ops);
```

### 需要知道的规则

- **匹配**：`ftrainOpsCreate` 把用户 Pattern 与库内受支持拓扑做结构同构匹配，忽略加入顺序；不匹配返回 `FTRAIN_STATUS_UNSUPPORTED`。
- **Plan 内容**：创建成功即保证至少一个可用 Primitive；primitive 0 是推荐默认。库里会返回全部可用实现（best-first），你按 workspace 预算遍历挑选。
- **生命周期**：Plan/Primitive 在创建后与源 Ops/Args 解耦；Tensor 内存不托管，异步执行期间须保持有效。
- **状态查询**：任何调用失败后 `ftrainGetLastStatus()` 返回线程局部的最近状态。

### 环境变量

进程级开关集中声明在 `env.h`：

| 变量 | 作用 |
|---|---|
| `FTRAIN_ENABLED_PRIMITIVES` / `FTRAIN_DISABLED_PRIMITIVES` | 按名字允许/禁止实现 |
| `FTRAIN_DISABLED_FINDERS` | 按名字禁用选择启发式 |
| `FTRAIN_DISABLE_CACHE` | 关闭选择缓存（调试用） |
| `FTRAIN_KDA_SAVE_QKN` / `FTRAIN_KDA_SAVE_H` / `FTRAIN_KDA_KEEP_STATE` / `FTRAIN_KDA_DEV` | KDA 调试开关，训练不需要，见第 1 节"KDA 调试开关" |
