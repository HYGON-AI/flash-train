# MoE 通信融合算子：原生源码接入

本仓库包含两个原生算子家族：`AGGroupGemmFamily` 和 `MoEReduceRSFamily`，
支持持久化会话、便捷 C 接口以及统一的 `Pattern → Ops → Args → Plan` 分阶段接口。
运行时、HIP 内核和汇编源码位于 `src/primitive/hygon/moe`，直接编入
`libflash_train`，构建和运行均不需要独立的外部算子包。
保留原来的 GEMM、IPC、信号和通信计算重叠流水线，不插入 Torch 运算。
没有新增 Python/PyTorch 绑定或 autograd。

公开 C API 统一声明在 `include/flash_train/flash_train.h`，使用时只需
`#include <flash_train/flash_train.h>`。MoE 的句柄、配置结构、便捷执行接口和
Pattern/Args 接口均由此入口管理，不再提供独立的 MoE 公开头文件。
`common.h` 继续承载既有通用类型和接口；内部实现头文件不作为公开 SDK 安装。

2026-11-09 公开头文件整理原样移动 MoE 声明，仅同步内部 include 和安装清单。
共享、静态及关闭 MoE 的构建和单测通过；新共享/静态 SDK 的 C 消费者运行通过，
C++ smoke 消费者编译通过。三种库文件及 12 个汇编代码对象均与整理前字节一致，
未修改 ABI 或执行代码；本轮未运行 GPU smoke 或性能测试。
冻结记录和验证结果位于 `build/moe-public-header-20261109/`。

### 源码目录

`src/primitive/hygon/moe/` 直接包含后端适配、运行时、AG/RS 实现及 `CMakeLists.txt`，
私有头文件、HIP 内核和汇编分别位于其 `include/`、`kernels/` 和 `asm/gfx936/`。
汇编别名生成、代码对象内嵌脚本和私有符号链接脚本集中在该目录的 `cmake/` 下。
其中 `backend.cpp` 将框架 session、张量视图和执行流适配到 AG/RS 实现，不表示额外路线。
不再额外嵌套实现目录；汇编代码对象现已嵌入库，不再需要单独部署资源目录。

2026-11-09 目录整理将 34 份文件原样上移，逐文件 SHA256 不变；重新构建的 12 个
汇编代码对象与整理前完全一致。共享、静态及关闭 MoE 的构建/单测，两个 SDK 的纯 C
消费者，以及搬迁后共享 SDK 的八卡全部 10 条路线 smoke 均通过。
本轮未重跑 16k/32k 大 shape 性能 A/B，不将 smoke 结果作为性能实测结论。
冻结文件清单、构建和测试记录位于 `build/moe-layout-20261109/`。

## 构建

只需要本仓库、DTK 工具链；启用测试时另需 GoogleTest：

```bash
cmake -S . -B build/moe-native -G Ninja \
  -DFTRAIN_PLATFORM_HYGON_HIP=ON \
  -DFTRAIN_HYGON_HIP_ARCHS=gfx936 \
  -DFTRAIN_ENABLE_MOE=ON \
  -DFTRAIN_BUILD_TORCH_BINDINGS=OFF \
  -DFTRAIN_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/opt/dtk \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DDTKROOT=/opt/dtk
cmake --build build/moe-native
ctest --test-dir build/moe-native --output-on-failure
cmake --install build/moe-native --prefix /path/to/flash-train/install
```

`FTRAIN_ENABLE_MOE` 默认开启；显式关闭时仍提供稳定 C API，创建算子返回
`FTRAIN_STATUS_UNSUPPORTED`。`ftrainMoeIsAvailable()` 查询编译时是否启用了
后端，不代表设备、形状、IPC 连通性或汇编资源已经验证成功。
当前仅开放 gfx936、BF16、单机 ETP，拒绝未验证的 gfx938。

### 汇编资源与安装

本仓库从 `.s` 源码构建 12 个 `.co`，不读取外部算子目录的预编译代码对象。
两个 NT ring-chunk 的符号别名在构建时从对应 a2a-chunk 源码生成，与原实现一致。
构建产物保留在构建树的 `src/primitive/hygon/moe/code_objects/gfx936`，随后由 CMake
生成只读、4096 字节对齐的字节数组，编入 `libflash_train.so` 或 `libflash_train.a`。
默认通过 `hipModuleLoadData()` 从内存加载，不查找相邻目录，不写临时文件。
SDK 和 wheel 不再安装独立 `.co`；静态链接后的可执行文件也自带这些代码对象。
旧构建或旧安装中残留的 `.co` 不会自动删除，但默认执行不读取它们。
默认路径中的 `embedded://gfx936/` 只是内部缓存标识，不是文件系统路径。
缓存仍复用原 module/function 句柄，不改变 kernel launch、机器码或通算重叠调度。

`FTRAIN_MOE_ASM_DIR` 保留为显式外部资源目录覆盖。`FTRAIN_MOE_FC2_ASM`、
`FTRAIN_MOE_FC2_NK_ASM` 和 `FTRAIN_MOE_RS_ASM_DIR_GFX936` 提供单独的调试覆盖，
这些显式覆盖仍通过文件加载，缺失或无效时不悄悄回退到内嵌版本。
正常运行无需设置。全部调优变量统一使用 `FTRAIN_` 前缀，旧前缀不再识别。
私有内核命名空间为 `ftrain::moe`，核心类型位于 `ftrain::moe::core`，
兼容内核实现位于 `ftrain::moe_compat`。内部头文件为 `moe_core.h` / `moe_core/`，
汇编文件和入口符号使用 `ftrain_` 前缀；内部头文件不作为公开 SDK 安装。
源码导入哈希及当前哈希见 `src/primitive/hygon/moe/source_manifest.json`；
清单中的文件路径相对于 `src/primitive/hygon/moe/`。
清单中的历史归档路径用于本机追溯，归档文件不随源码发布，也不参与构建。

MoE 后端类为 `FTrainMoeBackend`，由 `makeMoeBackend()` 创建。
`isMoeBackendEnabled()` 仅查询编译时是否启用 MoE，不承诺设备或配置可运行。
后端中的 `execution_context_` 表示核心执行上下文；`core::Tensor` 等类型通过
`namespace core = ftrain::moe::core` 引用。以上均为私有 C++ 实现，公开 C API 不变。

MoE 的通用配置、张量绑定、参数校验及持久 session 收拢在
`src/include/flash_train/family/moe/session.hpp` 和 `src/family/moe/session.cpp`；
其中 `MoeBackend` 是供 session 使用的抽象执行契约，不包含 HIP 实现。
Hygon 后端工厂声明位于 `src/include/flash_train/primitive/hygon/moe/backend.hpp`，
实现仍在 `src/primitive/hygon/moe/backend.cpp`。源码根目录不再保留单独的
MoE runtime 文件；此目录整理不改变公开 C API、Plan 执行流程、持久资源生命周期、
GEMM、内嵌代码对象或通算重叠调度。

2026-11-09 内部命名整理同步更新了声明、调用和私有符号隐藏规则。
共享、静态及关闭 MoE 的构建和单测，以及共享/静态 SDK 的纯 C 消费者均通过；
12 个完整汇编代码对象与整理前字节一致，115 个导出符号的名称及类型不变。
八卡 smoke 因外部 KFD context 在空闲准入阶段超时，未启动算子进程；
本轮没有获得新的八卡正确性或大 shape 性能结果，不沿用此前结果冒充本轮验证。
本轮冻结源码、替换审计、构建和测试记录位于 `build/moe-internal-naming-20261109/`。

历史外部后端的验证结果不能代替当前内置源码后端的验证结果；
当前实现应按本文的构建和多卡测试命令重新验证。

## 路线与布局

| 算子 | 路线 | 原始权重布局 |
|---|---|---|
| AG | `ag_nn_a2a`、`ag_nn_ring`、`ag_nn_ring_chunk` | `[E,K,N_local]` |
| AG | `ag_nt_a2a`、`ag_nt_a2a_chunk`、`ag_nt_ring_chunk` | `[E,N_local,K]` |
| AG | `ag_nt_a2a_chunk_32k`、`ag_nt_ring_chunk_32k` | `[E,N_local,K]` |
| RS | `rs_nn` | `[E,K_local,N]` |
| RS | `rs_nt` | `[E,N,K_local]` |

使用 `ftrainAGGroupGemmConfigInit()` 和 `ftrainMoEReduceRSConfigInit()` 初始化配置，
默认路线分别是 `ag_nn_ring_chunk` 和 `rs_nn`。布局由路线明确指定，不依据
方阵形状猜测，也不插入权重转置。`_32k` 要求 **K=32768**，不是 M=32768。
所有原生路线自身的其他限制继续生效；不做静默回退。

## 上下文与生命周期

1. 每个进程一个 rank，调用方先选择对应 HIP device；目前支持 2–8 个 rank，
   具体路线可能要求 ETP2/4/8 或 ETP8。所有参与者在整个生命周期内保持一致的
   创建、更新和执行顺序，不支持多个线程扮演同组的不同 rank。
2. `ftrainCollectiveContextCreate()` 记录 rank、world size、device 和 byte-allgather
   callback。callback 每次同步收集各 rank 的 `bytes` 字节，按 rank 顺序写入
   `world_size * bytes` 的接收区；不得抛异常跨过 C 边界。
3. `ftrainAGGroupGemmCreate()` / `ftrainMoEReduceRSCreate()` 是集体准备操作，
   创建持久化通信流、缓冲、IPC 映射和信号。首先检查各 rank 配置一致性。
   callback 只用于准备和参数更新，不代替执行期的 GPU 通信。
4. 创建成功后上下文句柄可以释放，算子会保留上下文；callback 的 `user_data`
   必须继续存活直到所有引用该上下文的算子和 Plan 销毁。
5. Execute 在传入流上异步提交；输入、输出、权重、路由和概率的设备存储均由
   调用方持有，不复制、不隐式转置、不转入 Torch。必须保持所有存储有效，直到
   使用它们的所有工作完成。不同流执行同一 session 时用事件建立依赖。
6. `ftrainAGGroupGemmSynchronize()` / `ftrainMoEReduceRSSynchronize()` 等待该 session
   的最近一次成功执行，而不是每次 Execute 都进行设备同步。失败清理可能同步设备。
   修改权重、路由、输入或回收内存前还需满足其他消费者的同步要求。
7. 销毁接口等待本地完成。**所有 rank 必须先完成并通过应用层 barrier，再集体
   释放算子和引用它的 Plan**，防止 peer 仍在访问 IPC 内存。销毁不是全组 barrier。
   必须在对应设备上销毁；一个 session 的调用不允许并发。

迁入的部分路线在 host dispatch 时使用进程级临时环境变量；本适配层对自己的
创建、更新和执行加进程级互斥锁。其他直接修改这些环境变量的代码不受这把锁保护，
应用仍需序列化与这些调用的冲突。锁不延伸到 GPU 执行期，不把流水线改成串行 GPU 工作。

## 输入输出契约

`FTrainMoeTensor` 包装一个连续设备 `FTrainStorageView`，附带 `identity` 和 `version`。
identity 为空时使用存储地址。权重或路由原地修改后，必须提升 version；不得为不同
内容复用同一个 identity/version。外部存储需要确实位于当前设备，分配大小和生命周期
由调用方负责；输出不可与其他输入重叠。

| 算子 | 配置字段 | 输入 | 路由 | 输出 |
|---|---|---|---|---|
| AG | `m_dim=full_m, n_dim=N_local, k_dim=K` | `[rank_m_sizes[rank],K]` | `[M,topk]` | `[M*topk,N_local]` |
| RS | `m_dim=ntokens, n_dim=N, k_dim=K_local` | `[M*topk,K_local]` | `[M,topk]` | `[rank_m_sizes[rank],N]` |

数据和权重为 BF16，路由为 INT32 或 INT64。RS 另外必须传入 BF16 的
`expert_weight[M,topk]`；不遗漏概率加权、还原 token 顺序及跨 rank 归约。
RS 输入使用本算子 ABI 的 token/choice 展开顺序；不能直接传另一个框架的 expert-major
缓冲而假定排序一致。AG native chunk/ring 使用各 rank 已展开、expert-major 的输入，
通常要求 operator topk=1。这里没有接管 EP All-to-All。

## 动态 M 与 Plan

两个算子的便捷 Execute 接口与 `ftrainGemm()` 一样，均经过统一的
`Pattern → Ops → Args → Plan → ExecutePrimitive` 流程，不再直接调用后端 session。
每个算子家族复用进程级不可变的 Pattern/Ops；每次执行重新绑定 Args、创建轻量 Plan，
再执行唯一的显式路线 Primitive。Primitive 复用创建算子时准备的持久化 session，
不重新初始化通信资源，不新增 GPU 同步，不改变 GEMM 或通算重叠调度。
便捷接口保留每个张量的 identity/version，空 identity 仍回退为存储地址。
需要避免每次创建 Args/Plan 的应用，可使用分阶段接口，在绑定不变时重复执行同一 Plan。

`capacity_m=0` 表示初始 M；需要增长时应显式预留容量。rank sizes 为空表示均分，
否则必须有 world_size 个正整数且和为 M。原 `ag_nt_a2a_chunk` 仅支持均分且每 rank
至少 8 行；不要将它与支持不均分的 `_32k` chunk 混淆。

SetRankMSizes 是集体更新：先完成本地执行，再通过全组 bootstrap barrier 更新，
不允许超过创建时容量。Execute 可每次传入不同地址、尺寸和版本，不需要重建通信资源。

分阶段接口先用 `ftrainPatternAddAGGroupGemm()` / `ftrainPatternAddMoEReduceRS()` 接线，
通过常规 `ftrainArgsSetTensor()` 绑定视图，再用 `ftrainArgsSetAGGroupGemm()` /
`ftrainArgsSetMoEReduceRS()` 关联已创建 session 与权重/路由版本，然后创建 Plan。
Finder 和 applicability 仅保留该显式路线，不重复初始化八个通信算子。
配置键覆盖路线、形状、分区和设备等元数据，不包含裸地址或 mutation version。

Plan 保留 session 和创建时的视图/版本，但不持有外部设备分配。
Args、Pattern 和原算子句柄释放后，Plan 仍保留必要状态。输入地址、形状或版本改变时
需要重新绑定 Args 并创建新 Plan；这只重新配置轻量 Primitive，复用原 session。
更新动态 M 后旧 Plan 不会自动更新，必须重新绑定后再执行。

## Python / PyTorch 接口

构建时启用 `FTRAIN_BUILD_TORCH_BINDINGS=ON`，或按 README 构建 wheel。
`flash_train.torch` 现在提供 `AGGroupGemm`、`MoEReduceRS` 和 `moe_is_available()`，
保留原 `gemm()`。核心 `libflash_train` 仍不链接 Torch；张量和通信组适配仅存在于
可选 Python 扩展及 `python/flash_train/torch/moe.py`。
HIP 头文件与 HCU Torch 的 CUDA 兼容头仍分翻译单元编译。

示例由 `torchrun --nproc-per-node=8 example.py` 启动：

```python
import os
import torch
import torch.distributed as dist
from flash_train.torch import AGGroupGemm, MoEReduceRS

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")

with AGGroupGemm(
    m_dim=128, n_dim=256, k_dim=1024, num_experts=8,
    capacity_m=256, route="ag_nn_ring_chunk",
) as ag:
    local_input = torch.ones((16, 1024), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((8, 1024, 256), dtype=torch.bfloat16, device="cuda")
    routing = (torch.arange(16, device="cuda") // 2).repeat(8).to(torch.int32)[:, None]
    output = ag(local_input, weight, routing)
    ag.synchronize()

with MoEReduceRS(
    m_dim=128, n_dim=4096, k_dim=1024, num_experts=8,
    topk=2, capacity_m=256, route="rs_nt",
) as rs:
    expanded = torch.ones((256, 1024), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((8, 4096, 1024), dtype=torch.bfloat16, device="cuda")
    routing = (torch.arange(128, device="cuda")[:, None] + torch.arange(2, device="cuda")) % 8
    probabilities = torch.full((128, 2), 0.5, dtype=torch.bfloat16, device="cuda")
    output = rs(expanded, weight, routing, probabilities)
    rs.synchronize()

dist.destroy_process_group()
```

该版本为 forward-only 绑定，没有 autograd/backward；梯度模式下传入 requires_grad
输入会显式报错，不能默默返回错误梯度。仅需前向时可用 `torch.no_grad()`。
所有 rank 必须在同一节点，同一 session 的创建、动态 M 更新和执行顺序必须一致。
若训练主通信组是 NCCL/RCCL，使用所有参与 rank 共同创建的 Gloo 子组并通过
`process_group=group` 传入；不要传 NCCL 组做 CPU bytes bootstrap。
Gloo 只交换初始化/动态 M 的 CPU 元数据，AG/RS 数据通信和通算重叠仍由原后端执行，
没有用 Torch collectives 替代 GPU 数据路径，也不执行权重转置或额外输入重排。

公共方法与参数：

- `operator(input, weight, routing, ..., out=None, stream=None, weight_version=None, routing_version=None)`：
  `out` 为空则分配输出，否则直接写入指定 BF16 连续缓冲；禁止与输入重叠。
- NN/NT 权重形状遵循前文原始布局；只按显式 `route` 选择，不自动转置或切换路线。
- 默认使用对应设备的当前 stream；显式 `stream` 必须属于该设备，输入准备就绪的
  stream/event 依赖由调用方建立。后端保留原有同 session 跨流完成事件依赖。
- 默认权重/路由版本取 Torch mutation counter，identity 使用 TensorImpl；原地 Torch
  更新后自动失效。外部内存写入或 inference tensor 应显式提供递增版本，不能重用旧值。
- 对参与执行的 Torch 分配调用 `record_stream`，并保留最近一次权重/路由引用，避免
  缓存身份被释放后复用。外部非 Torch 管理的设备内存仍由调用方保证生命周期。
- `set_rank_m_sizes(m_dim, rank_m_sizes=None)`：集体更新 M，保留原容量/路线限制；
  失败时 Python 端 M/partition 不变。不传 sizes 表示均分。
- `synchronize()` 等待当前 session；`close()` 等待并释放，可重复调用，也支持 `with`。
  一个实例不允许并发调用；应在销毁 process group 前显式关闭所有实例。

Python 测试与 GEMM 一样放在 `tests/python/`：

- `torch_moe_test.py`：导出、参数/版本/关闭状态/回调错误，以及十路线多卡入口。
- `moe_numerics_worker.py`：Torch CPU FP32 matmul 参考与 BF16 舍入；八卡验证 NN/NT、
  均分/不均分/恢复均分、同 M/同地址权重和路由原地更新、int32/int64 路由、显式输出/自动分配、跨流、
  inference tensor 显式版本和错误参数。原 chunk 路线仍只测其支持的均分变 M。

```bash
python3 -m pytest -q tests/python/torch_gemm_test.py tests/python/torch_moe_test.py -m 'not multirank'
FTRAIN_TEST_MOE_MULTIRANK=1 python3 -m pytest -q tests/python/torch_moe_test.py -m multirank
```

也接入 CTest：启用 bindings 和 tests 后注册 `ftrain_torch_api`；再启用
`FTRAIN_TEST_MOE_MULTIRANK=ON` 注册十个 `ftrain_torch_moe_<route>`，与 C++ 多卡测试
使用相同 GPU resource lock。Python 多卡 case 超时和失败时复用安全启动器清理本次
拥有的子进程。逐 rank 日志保留在构建目录 `tests/python-moe-numerics/<route>/run-*/`。
未显式启用时 pytest 跳过多卡项，不强制普通 Python 测试占八卡。
这些测试检验接口与小形状数值，不构成 Python 包装层性能无损或完整训练接入的结论。

## 验证与性能口径

### Python 层本机验证环境说明

本机 Torch 为 2.10.0/HIP 6.3。初始 `import torch` 会因系统
`/opt/hyhal/lib/libamd_smi.so` 缺少 `smi_init` 失败，独立于本项目绑定；
MIOpen 导出的 CMake target 还引用了不存在的 `/usr/lib/x86_64-linux-gnu/librt.so`。
本轮没有改系统安装或生产源码来掩盖这些问题，而是在被 Git 忽略的
`build/moe-python-20261109/env/` 中隔离处理：

- `sitecustomize.py` 仅让 Torch 跳过可选 amdsmi 遥测导入，不替代 GPU runtime 或算子。
- `miopen-config.cmake` 为本地构建将该导出依赖改用已存在的 `librt.so.1`。

因此本机复现 CTest 需保留临时环境目录：

```bash
PYTHONPATH="$PWD/build/moe-python-20261109/env${PYTHONPATH:+:$PYTHONPATH}" \
  ctest --test-dir build/moe-python-20261109/shared --output-on-failure
```

正常 Torch/DTK 安装不需要这两项临时处理；wheel 打包源码不包含这些环境补丁。
CTest 在运行时 prepend 构建出来的 Python package，保留调用者原有 PYTHONPATH，
不将本机验证环境路径写进项目源码。

### Python 接入验证结果

2026-11-09 使用重新构建的 Python 扩展完成以下验证：

- shared 完整 CTest 为 **24/24 通过**，包括原 GEMM/Python API、C/C++ 测试、
  无 Torch 核心依赖检查、十条原生八卡路线和十条 Python 八卡路线。
- 十条 Python 路线各完成 96 次逐 rank 输出检查，共 **960 次**；覆盖均分与不均分 M、
  权重和路由原地更新、显式输出与 stream，以及 inference tensor 的显式版本。
  原 `ag_nt_a2a_chunk` 仍按既有约束拒绝不均分 M，没有扩展核心路线的能力边界。
- static 常规 CTest 为 **4/4 通过**，另通过 `ag_nn_ring_chunk`、`rs_nt`
  两条 Python 八卡代表路线；扩展直接静态链接核心库。
- 关闭 MoE backend 后常规 CTest 为 **3/3 通过**；只有一张可见设备时，
  Python 八卡测试正确显示 Skipped，不将设备不足记为数值通过。

完整日志位于 `build/moe-python-20261109/` 下的 `final-all-ctest.log`、
`static-final-ctest.log`、`static-two-routes.log`、`disabled-final-ctest.log`
和 `python-preflight-skip.log`。逐 rank 输出记录位于对应构建目录的
`tests/python-moe-numerics/<route>/run-*/`。
这些结果仅证明本轮接口与数值回归通过，不代表性能无损，也不包含 autograd/backward。

### C/C++ 测试组织

MoE 正确性测试与原 GEMM 使用相同的分层方式；C++ API、family、primitive 和
数值用例均加入 `ftrain_test`，链接 `GTest::gtest_main`，不再维护独立数值测试 `main()`。
多卡数值用例通过启动器运行每个 rank 的同一 GoogleTest case，再接入 CTest。
普通单测没有多卡上下文时，这些数值用例显示 Skipped，不会启动额外进程或宣称完成数值校验。

| 位置 | 职责 |
| --- | --- |
| `tests/api/moe_api_test.cpp` | 公共接口的参数、默认路线、错误传播、版本/地址绑定、Plan 生命周期及快照 |
| `tests/api/moe_staged_fixture.hpp` | 公共 Pattern → Ops → Args → Plan 测试 fixture，统一释放句柄 |
| `tests/family/moe/family_test.cpp` | 两个 family 的 Pattern/全局注册、十条显式路线选择、Problem key、Primitive clone/workspace |
| `tests/family/moe/session_test.cpp` | 路线/布局/维度约束、动态分区事务性、张量 dtype/shape/设备内存/连续性/对齐校验 |
| `tests/primitive/hygon/moe/assets_test.cpp` | 内嵌 `.co` 字节、资源定位、覆盖目录及真实 HIP module/kernel 加载 |
| `tests/api/moe_numerics_test.cpp` | 参数化 GoogleTest 数值用例，以 CPU 参考分别验证便捷 API 与 Plan 的实际输出 |
| `tests/support/moe_session.hpp` | 无 GPU kernel 的 mock session/backend，供 API 和 family 测试复用 |
| `tests/support/run_multirank.py` / `run_multirank_test.py` | 多卡启动、GoogleTest JSON 校验、逐 rank 日志、超时/失败/信号清理及启动器自身的无 GPU 单测 |

`tests/api/pattern_api_test.cpp`、`tests/operation/` 和 `tests/ops_args_test.cpp`
也覆盖新增的两类 operation，不再只枚举原来的四类 GEMM。
MoE Python binding 的测试位于 `tests/python/torch_moe_test.py`，与原 GEMM Python 测试并列，
由 bindings 构建开关控制；无 Torch 的 C/C++ 测试仍可独立构建运行。
已有 `tests/moe/moe_perf.hip` 与 A/B 脚本仅用于性能验收，不计入数值单测。
统一性能入口现在也支持 `benchmarks/` 中的 `ag_group_gemm`、`moe_reduce_rs`，
Python/C/Plan/Primitive 四层均记录多卡样本；用法与计时口径见 `benchmarks/README.md`
的“MoE 多卡基准”。原 A/B 脚本保留用于历史交叉验证，不能将不同统计口径直接混比。
逐次运行的验证记录保留在本机构建目录，不作为源码或统一性能基线发布。

在已按前文配置的构建目录中运行默认测试：

```bash
cmake -S . -B build/moe-native -DFTRAIN_BUILD_TESTS=ON
cmake --build build/moe-native --target ftrain_test -j8
ctest --test-dir build/moe-native --output-on-failure
```

八卡数值测试默认不注册，显式启用后，每条路线对应一个 CTest case：

```bash
cmake -S . -B build/moe-native -DFTRAIN_TEST_MOE_MULTIRANK=ON
cmake --build build/moe-native --target ftrain_test -j8
ctest --test-dir build/moe-native --output-on-failure -R '^ftrain_moe_numerics_'
```

如果还要运行 Python 绑定的多卡数值测试，可使用 `-L numerics`，或按前文用 pytest
运行 `tests/python/torch_moe_test.py -m multirank`；Python 层仍与 GEMM 一样采用 pytest。

要求 `FTRAIN_ENABLE_MOE=ON`、Python 3 标准库、八张可见 gfx936，以及每卡至少
2 GiB 空闲显存。设备不足、架构不符或显存不足由 preflight 返回 77，CTest 显示
Skipped；真正的执行或数值错误返回失败，不能伪装成跳过。
每个测试限时 180 秒，某 rank 失败会立即结束本次测试拥有的其他进程组，
不会清理外部任务。CTest 通过同一 resource lock 串行使用 GPU。
逐 rank 日志与 GoogleTest JSON 保留在构建目录的 `tests/moe-numerics/<route>/run-*/`，
rendezvous 自动删除。启动器要求每个报告恰好包含选中的一个用例；过滤器未匹配、报告缺失、
数值断言失败或部分 rank 跳过均报告失败，不能把“未运行测试”统计为通过。

也可使用单路线入口，参数中的可执行文件改为统一的 `ftrain_test`：

```bash
bash tests/moe/run_smoke.sh "$PWD/build/moe-native/bin/ftrain_test" \
  ag ag_nn_ring_chunk "$PWD/build/moe-native/logs/ag_nn_ring_chunk" 8
bash tests/moe/run_smoke.sh "$PWD/build/moe-native/bin/ftrain_test" \
  rs rs_nn "$PWD/build/moe-native/logs/rs_nn" 8
```

底层等价于 `python3 tests/support/run_multirank.py <ftrain_test> ag|rs <route> <logs> --gtest`。
启动器先选择 `MoeNumericsEnvironmentTest.DevicesAvailable` 做设备检查，再为每个 rank
选择 `Routes/MoeNumericsTest.MatchesHostReference/<route>`。多卡参数通过测试私有环境变量
传递，GoogleTest 参数化提供路线；测试中的资源由 RAII 释放，数值比较复用
`tests/support/numerics.hpp` 的 `expectNear`，另外独立拒绝非有限输出。

安装包 consumer 如启用 `FTRAIN_CONSUMER_BUILD_SMOKE`，也构建 GoogleTest 版 `ftrain_test`，
需提供 GTest CMake package；默认只构建 C API consumer，安装库本身不依赖 GTest。

测试不依赖 Torch：独立 CPU 参考验证全部输出，每次执行前用 NaN 填充输出，
分别检查便捷 C API 和 Plan，避免后者覆盖前者的错误；两条路径使用不同 stream。
容差为 `0.002 + 0.02*abs(reference)`，单独拒绝非有限结果。
参考先按实际 BF16 输入/权重舍入，再模拟 GEMM 的 BF16 输出以及 RS 的概率加权/求和；
该混合容差用于 BF16 舍入和归约顺序差异，不代表任意形状的误差上界。

普通 AG 使用 M=128、N_local=256、K=1024、E=8、topk=1，
两条 `_32k` 路线仅将 K 改为 32768；RS 使用 M=128、N=4096、K_local=1024、
E=8、topk=2。每次 GEMM 的输入有 64 个有正负号的非零 K 位置，分散覆盖整个 K
（含首尾），权重覆盖所有 K；这不是稠密随机输入的完整压力测试。
各路线依次验证均分 → rank 间不均分 → 恢复均分，每阶段重复三轮；
同时覆盖同地址权重原地更新、变化的路由版本，以及 int32/int64 路由。
原 `ag_nt_a2a_chunk` 不支持不均分，因此中间阶段改为每 rank 从 16 行增到 24 行，
其不均分拒绝行为由 session 单测单独验证。NN/NT 使用原始对应权重布局，
AG 路由保持各来源 rank 内 expert-major 顺序，不引入测试专用权重转置。
文件 bootstrap 仅用于测试，不推荐用于生产。

安装后的消费验证可单独配置 `tests/moe/consumer`，仅通过
`find_package(flash_train)` 链接安装包。设置 `FTRAIN_CONSUMER_BUILD_SMOKE=ON`
还会构建相同的多卡测试程序，以验证资源定位与安装包的完整性。

这组小 shape 测试不用于证明 16k/32k 性能无损。正式性能验收仍需与迁移前后端
做同轮 A/B，保持 shape、路由输入、卡数和计时边界相同，记录 mean 和 median，
初始化、CPU 参考及验证不得混入稳态耗时。autograd/backward、训练框架具体接线
和大 shape 性能验收不在源码迁移完成的结论中。

### 2026-11-09 测试接入验证（GoogleTest 对齐前）

本节记录前一轮 C/C++ 测试接入：只修改测试、测试 CMake 和文档，
不修改生产算子、汇编、通信或调度；后续 Python 适配层的验证另行记录。
使用 `build/moe-family-layout-20261109/` 下的 shared、static、disabled 三个配置
重编译测试，最终结果如下：

| 配置 | GTest | CTest | 八卡数值 |
| --- | --- | --- | --- |
| shared | 204 通过，1 跳过 | 13/13 通过 | 全部 10 条路线通过 |
| static | 204 通过，1 跳过 | 13/13 通过 | 全部 10 条路线通过 |
| disabled | 198/198 通过 | 2/2 通过 | 后端关闭，不注册 |

shared/static 中跳过的是专门验证“后端关闭”的用例，并非数值测试。
每个配置的十路线八卡测试完成 1440 次独立输出校验
（10 路线 × 8 ranks × 3 阶段 × 3 repeats × 2 API 路径）。
启动器的 8 个无 GPU 单测通过，覆盖成功、skip、preflight 错误/超时、
找不到可执行文件、rank 失败、执行超时和信号中断。
另用 `HIP_VISIBLE_DEVICES=0` 验证八卡 case 在 CTest 中正确显示 Skipped。

将共享库安装到隔离前缀后，`tests/moe/consumer` 的 C API 程序和移位后的
数值 worker 均编译通过，安装包 `ag_nn_ring_chunk` 八卡数值通过。
原 `run_smoke.sh` 入口仍可使用。

测试修改前后的共享库/静态库逐字节一致，SHA256 分别为：

```text
shared  2cf093449b2e2e8df9421c6c765b8f4c76ee5b5b4b363b2f74e299b232f367c4
static  c77b6cb6dd7aa46a907c6f9f2cfd3d76095a9c15f36414c81ec779766c021be3
```

本机汇总日志和 GTest XML 在 `build/moe-tests-20261109/`，逐 rank 日志在
各构建目录的 `tests/moe-numerics/`。本轮是小 shape 正确性验证，
`_32k` 指 K=32768，不是完整 M=N=K=32768 的压力或性能验收。
构建、启动器、数值和依赖检查均通过；`git diff --check` 无错误。

### 2026-11-09 GoogleTest 对齐复验

将 `moe_numerics_test.cpp` 纳入 `ftrain_test`，移除原独立 `main()` 和
`ftrain_moe_smoke` 构建目标；现有 API/family/primitive 用例仍走同一个 GoogleTest
可执行文件。Python 绑定的 pytest 入口、统一性能 benchmark 和生产实现不变。

- 十条原生路线通过八卡 CTest，保留 80 份实际执行的 GoogleTest rank 报告，
  完成 1440 次全输出检查；三阶段动态 M、权重/routing 更新、两种 API 和两条 stream 均保留。
- 启动器 **15 个无 GPU 单测**通过，额外覆盖结构化报告缺失/损坏、空过滤结果、错误用例、
  报告中的失败、rank 跳过一致性及 GoogleTest 进程失败清理。
- 普通运行无多卡上下文时，设备检查和十个数值用例明确显示 Skipped；非法 rank 则失败，
  不会作为环境缺失跳过。
- MoE 关闭的测试目标仍可构建，且不注册上述数值用例；安装包 consumer 的 GoogleTest
  构建与用例发现通过。原生测试可执行文件仍不依赖 Torch/Python 动态库。
- 修改前后生产共享库 SHA256 完全相同；本轮不修改 kernel、汇编、路线或计时策略。

构建和汇总日志在 `build/moe-python-20261109/gtest-alignment/`，
其中 `native-audit.log` 审计实际执行的全部 rank 报告和输出检查数，
`launcher-unit.log`、`host-audit.log` 记录启动器和跳过/失败语义验证。
逐 rank 数值报告位于 `build/moe-python-20261109/shared/tests/moe-numerics/`。

额外的 Python 全量回归未完成：`ftrain_torch_moe_ag_nt_ring_chunk` 在 400 秒后超时，
后续路线未全部完成。上述 C++ 数值通过不能解读为 Python 全量回归也已通过。

### 源码迁移前后性能 A/B

`ftrain_moe_perf` 是独立 HIP/C++ 程序，不依赖 Torch。使用同一个可执行文件，
仅通过动态库搜索路径切换基线版本与当前内置源码版本：

```bash
cmake --build build/moe-native --target ftrain_moe_perf
python3 tests/moe/run_perf_ab.py \
  --binary build/moe-native/bin/ftrain_moe_perf \
  --baseline /path/to/baseline/lib/libflash_train.so \
  --candidate build/moe-native/lib/libflash_train.so \
  --balanced-topk false \
  --output build/moe-perf/run-unique-name
```

基线默认也是独立的 Flash Train 库。如果对比历史外部核心版本，需要另外传入
`--baseline-backend /path/to/backend.so`，以核验它仅被基线加载。测试记录全部已映射
共享库及其 SHA256，不再假设外部后端的固定文件名。

默认逐路线执行 baseline → candidate → candidate → baseline（ABBA），八卡、BF16、
E=64、逻辑 topk=6，预热 10 次、计时 40 次。六条普通 AG 和两条 RS 使用逻辑
M=N=K=16384；两条 `_32k` AG 使用 M=N=K=32768。AG 实际传入的 M 是逻辑 M 的
六倍，operator topk=1、N_local=N/8；RS operator topk=6、K_local=K/8。
NN/NT 直接使用各自的原始权重布局，不在计时内外转换成另一条路线。

测试生成确定性稠密 BF16 数据（非全零），AG 默认 balanced_topk=false、unique_topk=false，
使用确定性散列生成非强制均衡的 expert 路由，按 rank 内稳定 expert-major 顺序展开输入。
`--balanced-topk true` 可复现先前强制均衡的路由；日志记录标志和各 expert token 数，
不同标志的结果分别汇总，不能混为同一基线。RS 使用 token/choice 顺序及归一化概率。
这些数值不是历史 Torch 随机种子的复刻；本轮以所有 A/B 的设备输入指纹相同为准。
设置 `--partition ragged` 可测 rank 间不均分 M，但原 `ag_nt_a2a_chunk` 不适用。
正式启动前连续两次确认八卡空闲且没有外部 KFD context，默认最多等待 180 秒；
可用 `--idle-timeout` 调整。测试期间发现其他 GPU 进程时拒绝该次结果，不终止其他作业。

计时以用户流上的 HIP events 包围完整 Execute；每次等待结束事件，预热后采样。
初始化、数据生成、bootstrap、CPU 参考和全输出有限值检查不计时；算子内部预处理、
GEMM、通信及尾部处理仍计入。输出保存逐 rank 原始 GPU 时间、host enqueue 时间、
加载的库路径及 SHA256、输入指纹、数值验证和采样输出。

同时报告全部 rank 样本的 mean/median，以及每次取最慢 rank 后的 critical mean/median。
默认以 mean 和 critical mean 的增幅均不超过 3% 为噪声范围内通过；超出时标为 RECHECK，
不能仅凭一次抖动直接宣称回归。CPU 参考抽查 128 个输出，全输出扫描非有限值，
容差 `0.0005 + 0.025*abs(reference)`，并比较 A/B 的采样输出。

可用 `--cycles 2` 增加独立 ABBA 周期；`tests/moe/summarize_perf_ab.py` 可合并不同
运行目录的全部正式样本，同时保留每次启动等权统计，避免仅选择最快复测结果。
本机历史性能记录不随源码发布；正式性能结论应以相同输入、环境和统计口径的重新测量为准。
