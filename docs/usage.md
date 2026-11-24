# flash-train 使用指南

本库面向两类使用者，按接入成本从低到高排列。两类入口最终执行同一套引擎选择与内核调度，只是封装层次不同。

| 你是谁 | 用哪套接口 | 一次调用的成本 | 适合场景 |
|---|---|---|---|
| PyTorch 用户 | `flash_train.torch`（wheel） | 每次调用含参数组装与选择缓存查询 | 快速接入、原型验证 |
| C/C++ 调用方 | `common.h` 分阶段 API | Plan 复用后仅剩执行 | 训练热循环 |

每个算子在 `common.h` 里有一套专属声明：命名的参数结构体、专属的 Plan 句柄类型，以及以算子名开头的分阶段函数（创建、销毁、查询、执行）。`flash_train.h` 是伞头文件，当前只包含 `common.h`。

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

- 参数约束：`q` 为 **BF16、三维、连续、设备**张量 `(T,H,D)`，共享 `kv` 为 BF16 `(S,D)` 或 `(S,1,D)`，`out/dout` 为 BF16 `(T,H,V)`；`lse/attn_sink` 为 FP32，`topk_idxs/topk_length` 为 INT32，形状见第 2 节。所有 Tensor 须在同一设备，`T/H/S/K>0`，`D=512/576`、`V=512` 或 `D`；不满足抛 `RuntimeError`。
- 输出 `dq/dkv/dsink` 由库分配并返回；计算提交在 **torch 当前流** 上，与相邻算子自然保序。
- `softmax_scale` 是 Python 标量，按 FP32 传入内核，默认值为 1.0。

---

## 2. 分阶段 C API（common.h）

分阶段 API 把**一次性的选择与绑定**与**每次迭代的执行**分开，Plan 建好后在循环外复用：

```c
#include <flash_train/common.h>

/* ---- 每组缓冲区一次：选择实现并创建 Plan ---- */
FTrainSparseAttnBwdArgs args = {
    .q = q, .kv = kv, .out = out, .dout = dout, .lse = lse, .attn_sink = attn_sink,
    .topk_indices = topk_idxs, .topk_length = topk_length,
    .dq = dq, .dkv = dkv, .dsink = dsink,
    .softmax_scale = 0.04419417382415922F, .compute_type = FTRAIN_NUMERIC_TYPE_BF16};
FTrainSparseAttnBwdPlan plan;
FTrainStatus status = ftrainSparseAttnBwdCreate(&plan, &args);

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

workspace 大小以 `ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes` 的返回值为准（当前 BF16 内核为 `4*S*D+4` 字节，按 FP32 对齐）；输入、输出和 workspace 不得重叠。

参数结构体的每个字段就是算子的一个参数：

| 字段 | 要求 |
|---|---|
| `q`/`kv`/`out`/`dout`/`dq`/`dkv` | BF16 连续设备 Tensor 视图（`FTrainStorageView`），按 16 字节对齐；`q/dq` 为 `(T,H,D)`，`kv/dkv` 为 `(S,D)` 或 `(S,1,D)`，`out/dout` 为 `(T,H,V)` |
| `lse`/`attn_sink`/`dsink` | FP32 连续设备 Tensor 视图，按元素类型对齐；`lse` 为 `(T,H)`、仅选中 KV 且不含 sink 的 logsumexp，`attn_sink/dsink` 为 `(H)`；`out/lse` 须来自同一前向输入与选中集合 |
| `topk_indices`/`topk_length` | INT32 连续设备 Tensor 视图，按元素类型对齐，形状为 `(T,K)` / `(T)`；无效索引屏蔽、重复索引累加梯度；省略长度时仍提供 `(T)` 视图，仅将内存置为 `NULL`，否则长度限制到 `[0,K]` |
| `softmax_scale` | `float` 标量，创建 Plan 时按值复制 |
| `compute_type` | 计算精度，当前基础实现为 `FTRAIN_NUMERIC_TYPE_BF16` |

`ftrainSparseAttnBwdCreate` 返回 `FTRAIN_STATUS_INVALID_ARGUMENT`（`plan`/`args` 为空、compute 枚举非法、视图或形状不一致）、`FTRAIN_STATUS_OVERFLOW`（张量字节溢出）、`FTRAIN_STATUS_UNSUPPORTED`（无实现满足参数）；`ftrainSparseAttnBwdExecute` 返回 `FTRAIN_STATUS_INVALID_ARGUMENT`（下标越界、设备不符、workspace 不足）。失败详情可用 `ftrainGetLastStatus()` / `ftrainGetLastMessage()` 查询。

### 需要知道的规则

- **Plan 内容**：创建成功即保证至少一个可用 Primitive；primitive 0 是推荐默认。库里会返回全部可用实现（best-first），可用 `ftrainSparseAttnBwdGetNumPrimitives`、`ftrainSparseAttnBwdGetPrimitiveName`、`ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes` 按下标遍历，并按 workspace 预算挑选。
- **绑定**：Plan 绑定创建时的缓冲区地址、`softmax_scale` 的取值和调用线程的当前设备。换缓冲区、改 `softmax_scale` 或换设备都要重新创建 Plan。
- **生命周期**：Plan 创建后独立存活，不依赖调用方传入的 `args` 及其指向的维度数组，`ftrainSparseAttnBwdExecute` 返回后即可销毁。函数不同步：Tensor 内存不托管，须保持有效直至该流上的工作完成。
- **状态查询**：任何调用失败后 `ftrainGetLastStatus()` 返回线程局部的最近状态，`ftrainGetLastMessage()` 返回对应信息。

### 环境变量

进程级开关集中声明在 `env.h`：

| 变量 | 作用 |
|---|---|
| `FTRAIN_ENABLED_PRIMITIVES` / `FTRAIN_DISABLED_PRIMITIVES` | 按名字允许/禁止实现 |
| `FTRAIN_DISABLED_FINDERS` | 按名字禁用选择启发式 |
| `FTRAIN_DISABLE_CACHE` | 关闭选择缓存（调试用） |
