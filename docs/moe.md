# AGGroupGemm 通信融合算子

AGGroupGemm 已接入仓库原有的 Pattern、Ops、Args、Plan 和 Primitive 框架。
核心库不依赖 Torch；Python / PyTorch 绑定是可选适配层。

## 代码组织

- `src/api/moe_api.cpp`：C API 与框架接线。
- `src/family/moe/`：问题描述、显式路线选择和持久 session 管理。
- `src/primitive/hygon/moe/`：设备后端、公共运行时、通信、HIP 内核和汇编资源。
- `python/flash_train/torch/moe.py`：可选 Python 适配。

通信上下文、张量封装、内存管理、路由元数据及资源加载可供其他通信融合算子复用。
当前仅注册 AGGroupGemm；本提交不提供 MoEReduceRS 的公开接口或实现。

## 构建

```bash
cmake -S . -B build/moe -G Ninja \
    -DFTRAIN_PLATFORM_HYGON_HIP=ON -DFTRAIN_HYGON_HIP_ARCHS=gfx936 \
    -DFTRAIN_ENABLE_MOE=ON -DFTRAIN_BUILD_TESTS=ON \
    -DFTRAIN_BUILD_BENCHMARKS=ON
cmake --build build/moe -j8
ctest --test-dir build/moe --output-on-failure -LE multirank
```

`FTRAIN_ENABLE_MOE=OFF` 时仍可编译 SDK，执行入口返回不支持。
`FTRAIN_BUILD_TORCH_BINDINGS=ON` 用于显式构建可选 PyTorch 绑定。
汇编在构建时生成 code object 并嵌入库，默认不需要外部 `.co` 文件。
`FTRAIN_MOE_ASM_DIR` 可显式覆盖汇编资源目录，主要用于调试。

## 路线与布局

| 路线 | 原始权重布局 |
| --- | --- |
| `ag_nn_a2a` | `[E,K,N_local]` |
| `ag_nn_ring` | `[E,K,N_local]` |
| `ag_nn_ring_chunk` | `[E,K,N_local]` |
| `ag_nt_a2a` | `[E,N_local,K]` |
| `ag_nt_a2a_chunk` | `[E,N_local,K]` |
| `ag_nt_ring_chunk` | `[E,N_local,K]` |
| `ag_nt_a2a_chunk_32k` | `[E,N_local,K]` |
| `ag_nt_ring_chunk_32k` | `[E,N_local,K]` |

默认路线为 `ag_nn_ring_chunk`，不根据权重形状自动切换 NN / NT。
`_32k` 路线要求 K=32768，不是 M=32768。
NT chunk / ring 路线要求八卡；原始 `ag_nt_a2a_chunk` 要求 rank 间均分 M。
其他路线的设备数、对齐及 ragged-M 限制由配置校验报告。

## C API 与资源管理

公共类型和分阶段接口声明位于 `include/flash_train/common.h`，
便利执行入口位于 `include/flash_train/flash_train.h`。

1. 通过 `ftrainCollectiveContextCreate()` 创建通信上下文，提供 rank、设备和 CPU 字节 AllGather 回调。
2. 使用 `FTRAIN_AG_GROUP_GEMM_CONFIG_INIT` 初始化配置，填写维度、expert 数及路线。
3. 调用 `ftrainAGGroupGemm(context, &config, input, weight, routing, output, stream)`。
4. 使用 `ftrainCollectiveContextSynchronize()` 等待完成，最后销毁上下文。

所有 rank 的配置和首次调用顺序必须一致。上下文按结构配置缓存 session，
重复执行不重新初始化通信和计算资源。动态 M 应固定 `capacity_m` 并更新
`m_dim`、`rank_m_sizes` 和 `num_rank_m_sizes`；零 capacity 默认归一化为当前 M。

输入为 `[rank_m_sizes[rank],K]`，routing 为 `[M,topk]`，
输出为 `[M*topk,N_local]`。数据、权重和输出使用 BF16，routing 使用 INT32 / INT64。
除 `ag_nn_a2a` 外的路线使用已展开的输入，算子 topk=1。
`FTrainMoeTensor` 携带张量视图、identity 和 version，输入修改后应正确更新版本。

分阶段调用使用 `ftrainPatternAddAGGroupGemm()` 和 `ftrainArgsSetAGGroupGemm()`，
之后走原有 `ftrainPlanCreate()`、`ftrainPlanExecutePrimitive()`。
参数绑定的 stream 用于首次资源准备，执行流由 Plan 执行接口指定。

## Python 适配

`flash_train.torch` 导出 `AGGroupGemm` 和 `moe_is_available()`。
调用前需要初始化 `torch.distributed`，通信组用于交换初始化元数据，
算子数据通信和通算重叠由设备后端执行。使用 context manager 或显式 `close()` 管理资源。

## 正确性和性能测试

- C++ GoogleTest：`tests/api/moe_api_test.cpp`、`tests/api/moe_numerics_test.cpp`。
- Python pytest：`tests/python/torch_moe_test.py`。
- 原生性能 worker：`tests/moe/moe_perf.hip`，A/B 编排在 `tests/moe/run_perf_ab.py`。
- 统一基准入口：`benchmarks/` 下的 `ag_group_gemm` 注册项。

启用 `FTRAIN_TEST_MOE_MULTIRANK=ON` 注册各路线的八卡正确性测试：

```bash
ctest --test-dir build/moe --output-on-failure -R '^ftrain_moe_numerics_'
```

统一基准的 shape 是全局逻辑 M、K、N。AG 在计时前按逻辑 topk 展开并按 rank / expert 排序，
算子 M 为逻辑 M×topk，N 为全局 N/TP，算子 topk=1。
计时包含 AllGather 与 grouped GEMM，参考计算和输出对齐不计入算子时间。
统一使用 `balanced_topk=False`、`unique_topk=False`，并同时记录 mean 与 median。
