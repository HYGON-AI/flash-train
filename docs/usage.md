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

---

## 4. 可选分布式算子（HCU wheel）

`AGGroupGemm` 和 `MoEReduceRS` 是**会话型**分布式算子：先在每个 rank 创建同一个 session，再绑定本 rank 的 tensor 视图创建 Plan。当前 vendor 内核要求 `nnodes=1`；session 必须在各 rank 上 collective 创建，shape/dtype/route/variant 在 Plan 生命周期内保持不变。

Hygon 分布式后端的 IPC 状态和 route 状态是进程级的。测试或多路线复用时，**每条 route 使用一个独立 `torchrun` 进程**。

Python API 使用 `init_hygon_shm` 和 `HygonDistributedPlan`。
wheel 内的 code object 位于 `flash_train/torch/hygon_assets`，可用
`FTRAIN_HYGON_CODE_OBJECT_DIR` 显式指定替代目录。
运行期调优变量使用 `FTRAIN_HYGON_<NAME>`；vendored C++ 原生读取该名字，
并保留旧诊断变量作为 fallback。若同时显式设置新旧两个名字，Hygon 名优先。
route 内部临时 override 会清理两个名字并恢复调用前现场。

### AGGroupGemm

可选 route：`native_nk_a2a`、`native_kn_a2a`、`native_kn_ring_fixed`、`native_kn_ring`、`native_kn_ring_readymask`。

```python
flash_train.torch.init_hygon_shm(etp_group)
session_id = flash_train.torch.create_ag_group_gemm_session(
    etp_group=etp_group,
    nnodes=1,
    full_m=global_expanded_rows,
    n_dim=n_per_rank,
    k_dim=k_dim,
    num_experts=num_experts,
    topk=operator_topk,
    dtype=torch.bfloat16,
    rank_m_sizes=rank_m_sizes,       # ragged/session 明确要求时提供；均匀分区可传 []
    capacity_full_m=-1,
    route="native_nk_a2a",
)
plan = flash_train.torch.create_ag_group_gemm_plan(
    session_id, input, weight, topk_ids, output
)
plan.execute()                        # 在当前 stream 上异步执行
...
flash_train.torch.close_ag_group_gemm_session(session_id)
```

输入/routing 的分布式 scope 由 route 决定；`output` 始终是**全局展开行数**（`native_nk_a2a` 为 `full_m * topk`，native-KN 展开路线为 session `full_m`）。

### MoEReduceRS

可选 route：`a2a`、`a2a_ragged_signal`；可选 GEMM variant：`nk_direct`、`kn_direct`。

```python
flash_train.torch.init_hygon_shm(etp_group)
session_id = flash_train.torch.create_moe_reduce_rs_session(
    etp_group=etp_group,
    nnodes=1,
    max_ntokens=max_ntokens,
    n_dim=n_per_rank,
    k_dim=k_per_rank,
    num_experts=num_experts,
    topk=topk,
    dtype=torch.bfloat16,
    rank_m_sizes=rank_m_sizes,
    n_chunks=0,
    route="a2a_ragged_signal",
    gemm_variant="nk_direct",
)
plan = flash_train.torch.create_moe_reduce_rs_plan(
    session_id, input, weight, chosen_experts, expert_weight, output
)
plan.execute()
...
flash_train.torch.close_moe_reduce_rs_session(session_id)
```

权重布局必须与 variant 匹配：`nk_direct` 使用 `[E, N, K_local]`，`kn_direct` 使用 `[E, K_local, N]`。完整依赖和生命周期规则见 [ADR 0001](adr/0001-hygon-distributed-backend.md)。
