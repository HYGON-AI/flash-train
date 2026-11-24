# flash-train 使用指南

本库面向三类使用者，按接入成本从低到高排列。三类入口最终执行同一套引擎选择与内核调度，只是封装层次不同。

| 你是谁 | 用哪套接口 | 一次调用的成本 | 适合场景 |
|---|---|---|---|
| PyTorch 用户 | `flash_train.torch`（wheel） | 每次调用含参数组装与选择缓存查询 | 快速接入、原型验证 |
| C/C++ 调用方（图省事） | `flash_train.h` 便利 API | 同上 | 偶发调用、不建 Plan 的场景 |
| C/C++ 调用方（追求性能） | `common.h` 分阶段 API | Plan 复用后仅剩执行 | 训练热循环 |

每个算子有一套专属声明：以算子名开头的分阶段函数（创建、销毁、查询、执行）和专属的 Plan 句柄类型在 `common.h`，便利函数在 `flash_train.h`。

---

## 1. PyTorch 用户

### 安装

在 HCU 环境（DTK + PyTorch）中从源码构建 wheel 并安装：

```bash
pip wheel . --no-deps -w dist
pip install --no-deps dist/flash_train-*.whl
```

### 计算 sparse_attn_bwd（BF16）

用 Torch 组合前向准备 `out/lse`，再显式调用 DSA 稀疏注意力反向：

```python
import torch
import flash_train.torch

T, H, S, D, K = 8, 8, 8, 512, 17
q = (torch.randn(T, H, D, device="cuda") * 0.1).to(torch.bfloat16)
kv = (torch.randn(S, D, device="cuda") * 0.1).to(torch.bfloat16)
sink = torch.zeros(H, device="cuda", dtype=torch.float32)
indices = (torch.arange(K, device="cuda", dtype=torch.int32) % S).expand(T, -1).contiguous()
scale = D ** -0.5
keys = kv.float()[indices.long()]
scores = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
lse = scores.logsumexp(-1)
logits = torch.cat([scores, sink[None, :, None].expand(T, -1, -1)], -1)
probabilities = logits.softmax(-1)[..., :-1]
out = torch.einsum("thk,tkd->thd", probabilities, keys).to(torch.bfloat16)
dout = torch.randn_like(out)
dq, dkv, dsink = flash_train.torch.sparse_attn_bwd(
    q, kv, out, dout, lse, sink, indices, softmax_scale=scale
)
```

要点：

- 参数约束：`q` 为 **BF16、三维、连续、设备**张量 `(T,H,D)`，共享 `kv` 为 BF16 `(S,D)` 或 `(S,1,D)`，`out/dout` 为 BF16 `(T,H,V)`；`lse/attn_sink` 为 FP32，`topk_idxs/topk_length` 为 INT32，形状见第 3 节。所有 Tensor 须在同一设备，`T/H/S/K>0`，`D=512/576`、`V=512` 或 `D`；不满足抛 `RuntimeError`。
- 输出 `dq/dkv/dsink` 由库分配并返回；计算提交在 **torch 当前流** 上，与相邻算子自然保序。
- `softmax_scale` 是 Python 标量，按 FP32 传入内核，默认值为 1.0。

---

## 2. 便利 C API（flash_train.h）

一个函数覆盖"选实现→执行"全过程，等价于第 3 节的分阶段流程写一遍样板：

```c
#include <flash_train/flash_train.h>

FTrainStatus status = ftrainSparseAttnBwd(
    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length,
    dq, dkv, dsink, softmax_scale, FTRAIN_NUMERIC_TYPE_BF16,
    workspace, workspace_bytes, stream);
```

| 参数 | 要求 |
|---|---|
| `q`/`kv`/`out`/`dout`/`dq`/`dkv` | BF16 连续设备 Tensor 视图（`FTrainStorageView`），按 16 字节对齐；`q/dq` 为 `(T,H,D)`，`kv/dkv` 为 `(S,D)` 或 `(S,1,D)`，`out/dout` 为 `(T,H,V)` |
| `lse`/`attn_sink`/`dsink` | FP32 连续设备 Tensor 视图，按元素类型对齐；`lse` 为 `(T,H)`、仅选中 KV 且不含 sink 的 logsumexp，`attn_sink/dsink` 为 `(H)`；`out/lse` 须来自同一前向输入与选中集合 |
| `topk_idxs`/`topk_length` | INT32 连续设备 Tensor 视图，按元素类型对齐，形状为 `(T,K)` / `(T)`；无效索引屏蔽、重复索引累加梯度；省略长度时仍提供 `(T)` 视图，仅将内存置为 `NULL`，否则长度限制到 `[0,K]` |
| `softmax_scale` | FP32 宿主机标量 Tensor（秩 0），创建 Plan 时按值复制 |
| `compute_type` | 计算精度，当前基础实现为 `FTRAIN_NUMERIC_TYPE_BF16` |
| `workspace`/`workspace_bytes` | 推荐实现所需 workspace（当前 BF16 内核为 `4*S*D+4` 字节，按 FP32 对齐，查询方式见第 3 节）；输入、输出和 workspace 不得重叠 |
| `stream` | 执行流；函数不同步，返回后设备内存及 workspace 须保持有效直至该流完成 |

状态码语义与第 3 节完全一致：`FTRAIN_STATUS_INVALID_ARGUMENT`（参数不一致/workspace 不足）、`FTRAIN_STATUS_UNSUPPORTED`（无实现满足参数）、`FTRAIN_STATUS_SUCCESS`。失败详情可用 `ftrainGetLastStatus()` / `ftrainGetLastMessage()` 查询。

---

## 3. 分阶段 C API（common.h）

分阶段 API 把**一次性的选择与绑定**与**每次迭代的执行**分开，Plan 建好后在循环外复用：

```c
#include <flash_train/common.h>

/* ---- 每组缓冲区一次：选择实现并创建 Plan ---- */
FTrainStorageView views[12] = {
    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length,
    dq, dkv, dsink, softmax_scale};
FTrainSparseAttnBwdPlan plan;
FTrainStatus status = ftrainSparseAttnBwdCreate(
    &plan, views[0], views[1], views[2], views[3], views[4], views[5], views[6], views[7],
    views[8], views[9], views[10], views[11], FTRAIN_NUMERIC_TYPE_BF16);

/* ---- 查询推荐实现的 workspace ---- */
uint64_t bytes;
ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes);
void* workspace;  hipMalloc(&workspace, bytes);

/* ---- 每次迭代：执行推荐默认实现 ---- */
ftrainSparseAttnBwdExecute(plan, 0, workspace, bytes, stream);

/* ---- 所有异步使用完成后释放 ---- */
hipStreamSynchronize(stream);
hipFree(workspace);
ftrainSparseAttnBwdDestroy(plan);
```

### 需要知道的规则

- **Plan 内容**：创建成功即保证至少一个可用 Primitive；primitive 0 是推荐默认。库里会返回全部可用实现（best-first），可用 `ftrainSparseAttnBwdGetNumPrimitives`、`ftrainSparseAttnBwdGetPrimitiveName`、`ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes` 按下标遍历，并按 workspace 预算挑选。
- **绑定**：Plan 绑定创建时的缓冲区地址、宿主机标量的取值和调用线程的当前设备。换缓冲区、改 `softmax_scale` 或换设备都要重新创建 Plan。
- **生命周期**：Plan 创建后独立存活，不依赖调用方传入的描述数组。Tensor 内存不托管，异步执行期间须保持有效。
- **状态查询**：任何调用失败后 `ftrainGetLastStatus()` 返回线程局部的最近状态，`ftrainGetLastMessage()` 返回对应信息。

### 环境变量

进程级开关集中声明在 `env.h`：

| 变量 | 作用 |
|---|---|
| `FTRAIN_ENABLED_PRIMITIVES` / `FTRAIN_DISABLED_PRIMITIVES` | 按名字允许/禁止实现 |
| `FTRAIN_DISABLED_FINDERS` | 按名字禁用选择启发式 |
| `FTRAIN_DISABLE_CACHE` | 关闭选择缓存（调试用） |
