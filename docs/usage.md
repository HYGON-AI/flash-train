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

## 诊断日志

日志由 C++ 核心统一输出到 `stderr`。C API 与 PyTorch 调用进入核心后使用同一机制；Python 绑定的前置检查仍由绑定自身报错。

在启动进程前配置：

```bash
FTRAIN_LOG_LEVEL=DEBUG python your_script.py
FTRAIN_LOG_LEVEL=INFO ./your_program 2>flash-train.log
```

`FTRAIN_LOG_LEVEL` 支持 `OFF`、`ERROR`、`WARNING`（默认）、`INFO`、`DEBUG`，包含该级别及更严重的信息。允许大小写差异和首尾空白；未设置/空值为 WARNING，非法值回退 WARNING 并仅尝试提示一次。配置在首次使用日志时读取，进程内固定；设置 OFF 也不会改变错误返回。

| 级别 | 内容 |
|---|---|
| ERROR | 原异常转状态码边界的实际失败 API、状态码和错误消息 |
| WARNING | 非法日志配置等提醒；普通缓存 miss、候选拒绝不是警告 |
| INFO | 自然读取的生效配置、重新选择的有序名单及推荐项 |
| DEBUG | 参数摘要、缓存、Finder、候选过滤/拒绝和实际执行请求/返回 |

输出形式：`[带时区时间][ftrain][级别][pid=...][thread=...][模块] key=value ...`。类型/布局的数字值对应 C SDK 中的枚举；不输出张量内容和裸地址。每条记录最多 4096 字节（含预留的换行/终止空间），控制字符会转义，非法 UTF-8 会替换，超长记录带 `...[truncated]`。候选和配置名单有界展开，配置集合不为日志重新排序。

### 如何解释事件

- `event=selection` 只说明候选名单和推荐第 0 项已确定，不代表 clone/configure 或 Plan 创建成功。
- 缓存命中输出 `state=hit`，不重复输出 INFO 选择结果，也不补造本次未发生的 Finder/适用性检查。缓存关闭时每次重选，INFO 可能较多。
- `event=execute_request` 表示调用方请求的实际实现，可能不同于推荐项；后续 workspace 检查仍可能失败。
- `event=execute_return` 仅表示执行接口返回，不保证提交过 kernel（例如空输出），不表示 GPU 完成，也不保证已发现异步设备错误。
- ERROR 归属实际异常转换接口，例如 `ftrainPlanCreate`，不一定是外层 `ftrainGemm`。便利入口转发、清理、last-result 恢复和查询不重复打印。
- 一次性初始化失败在实际失败时记录；后续仅返回保存的初始化失败状态，不保证再次产生新 ERROR。独立的新失败不按消息永久去重。

### 性能与可靠性边界

OFF/默认 WARNING 仍有级别判断和首次初始化开销；未启用的摘要和日志参数不求值。INFO/DEBUG 的格式化、输出可能显著增加 host 延迟。没有新增 GPU 查询、同步或计时。

同一进程本模块的记录通过独立锁写出，但不管理其他库/独立进程的 stderr，不保证跨进程顺序或记录原子性。PID 不是分布式 rank，第一版不按 rank 过滤，不提供调用 ID 关联。

日志异常和可检测写出失败按最佳努力丢弃，不修改业务状态/消息或 errno，不递归报告日志故障；同步 stderr 可能阻塞，进程信号和终止不能由 C++ 异常保护全部屏蔽。不安装信号处理、不改变全局缓冲策略、不处理多线程 fork 后的锁修复；正常多进程使用应分别初始化。没有文件轮转、后台线程、动态级别接口或 Python 独立 logger。原 ROCTX trace 保持独立。
