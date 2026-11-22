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

for t in (q, k, v, g, beta, A_log, dt_bias): t.requires_grad_()      # 需要梯度的输入
o = ft.kimi_delta_attention(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu)
o.sum().backward()                                      # 支持 autograd：q/k/v/g/beta/A_log/dt_bias 的梯度
```

要点：

- 布局为打包 THD：`q/k/v/o` 是 `[T, H, 128]` 的 BF16；`cu_seqlens`（int32 `[N+1]`）把 T 行切成互不相关的序列，每条序列从零状态开始，长度任意（不必是 64 的倍数，可以为 0）。省略时视为一条序列。
- 两种 gate 契约：
  - 传 `A_log`/`dt_bias` 时，g 是原始 gate，算子内部计算 `lower_bound·sigmoid(exp(A_log)·(g+dt_bias))`，用于 Kimi K3，`lower_bound` 取 [-5, 0)；
  - 不传时，g 是 FP32 的逐 token log 衰减（cuDNN-frontend PR #556 契约），要求 ≥ -5；`check_decay=True`（默认）会检查这一点。
- `use_qk_l2norm_in_kernel=True`（默认）表示在算子内部对 q/k 做 l2 归一化，eps 为 1e-6。
- 当前实现为手写 HIP 内核（`G2KdaForward`/`G2KdaBackward`），gfx936 与 gfx938 共用一套源码、运行时按架构分派（gfx938 另有专属优化路径）；只支持 K = V = 128，其他配置抛 `RuntimeError`（`FTRAIN_STATUS_UNSUPPORTED`）。确定性：只用固定顺序归约，不用原子操作。
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

`flash_train.torch.chunk_kda` 的签名与 FLA 的 `fla.ops.kda.chunk_kda` 相同，输入为 `[B, T, H, 128]`，返回 `(o, None)`。Megatron 侧只需要换 import，调用参数不变，**不需要设置任何 flash-train 的环境变量**。

KDA 的 HIP kernel（G2）已经编译在 flash-train 里（`libflash_train.so`），**不需要另外安装 hip_kda**。

**完整步骤速览**（每一步的细节见下面对应小节）：

1. 安装 wheel：镜像与预编译 wheel 一致时直接 `pip install`（1a）；否则在训练镜像里从源码编译（1b，约 3–4 分钟）；
2. 把 Megatron 里的 `from fla.ops.kda import chunk_kda` 换成 `from flash_train.torch import chunk_kda`，参数不动；
3. 照原来的命令启动训练，用打印或 profiler 确认 KDA 走的是 flash-train。

**环境要求**：gfx936（BW1000）与 gfx938（BW1100 / BW1102 / BW1300）；K = V = 128（K3 满足）。**同一个 wheel 同时支持两种架构**：CMake 按 `gfx936;gfx938` 编成一个胖包，运行时按 `gcnArchName` 自动分派——gfx936 走 mk3 基线，gfx938 走其专属优化（`__gfx938__` / 运行时判断包起来）。调用方不需要区分卡型，也不需要装两个包。CP 的验证在 `megatron:0.19.2-ubuntu22.04-dtk2610-py3.12-1009` 镜像（DTK 26.10、torch 2.11、Python 3.12）里做。

#### 输入形状：B / T / H 的含义

`[B, T, H, 128]` 沿用 FLA `chunk_kda` 的约定，和“T = 总 token 数”的常见写法不完全一样：

| 记号 | 含义 |
|---|---|
| B | 张量的 batch 维 |
| T | 每条样本的序列长度，不包含 B |
| H | 本卡上的 KDA head 数，即总 head 数 / TP（K3 为 96/TP，TP8 时 H = 12） |
| 128 | head_dim（K = V = 128） |

q / k / v / g 都是 `[B, T, H, 128]`，beta 是 `[B, T, H]`，A_log 是 `[H]`，dt_bias 是 `[H*128]`（或 `[H, 128]`），所以 B / T / H 确定后所有输入的形状都确定了。

两种输入方式：

- **定长（不传 `cu_seqlens`）**：B = micro batch size（`--micro-batch-size`），T = seq_len，每卡总 token 数 = B × T。例如 micro batch size 1、seq_len 32768、TP8 时，每卡输入为 `[1, 32768, 12, 128]`。
- **变长序列（packed / THD，传 `cu_seqlens`）**：一个 micro batch 里长度不同的多条序列首尾拼成一条，B 必须为 1，T = 拼起来的总 token 数（约 micro batch size × seq_len），`cu_seqlens`（int32 `[N+1]`）标出每条序列的边界，例如 `[0, 1000, 4000, 8192]`。所以在变长序列这种输入方式下，T 就是“总 token 数”。

预训练里如果只是把多篇文档直接拼接、没有传 `cu_seqlens`，对 KDA 来说仍是一条定长序列，属于第一种方式。

需要我们按实际训练配置测性能时，提供以下几项即可：TP（或每卡 head 数）、micro batch size、seq_len、是否传 `cu_seqlens`（是的话附大致的文档长度分布或一份典型的 `cu_seqlens`）、是否开全层重算、CP（context parallel）大小。

#### 1. 安装

wheel 绑定编译时的 DTK、torch 和 Python 版本：镜像与预编译 wheel 一致时直接安装（1a）；其他镜像在训练用的镜像（容器）里从源码编译（1b），换了镜像要重新编译。

**1a. 预编译 wheel**（megatron 0.19.2 / DTK 26.10 / torch 2.11 / Python 3.12；一个胖包同时支持 gfx936 + gfx938）

用从本 `zhipu` 分支统一出的 wheel（`flash_train-0.1.0-cp312-cp312-linux_x86_64.whl`，编译目标 `gfx936;gfx938`，运行时自动分派，支持 CP > 1）。镜像与 wheel 一致时直接安装：

```bash
python3 -m pip install --no-deps --force-reinstall flash_train-0.1.0-cp312-cp312-linux_x86_64.whl
python3 -c "import flash_train.torch as ft; print(ft.__file__, ft.chunk_kda)"
```

> 包由维护者从 `zhipu` 分支统一构建后分发；拿到包时请记下它对应的提交（PR 里会注明），反馈问题时附上。

**1b. 源码编译**

```bash
git clone -b zhipu https://github.com/HYGON-AI/flash-train.git
cd flash-train
git rev-parse --short HEAD                    # 记下实际使用的版本，反馈问题时附上

export DTKROOT=/opt/dtk                       # CMake 靠它找到 roctx
python3 -m pip install pybind11 scikit-build-core
# 镜像缺 libc6-dev 的开发链接时，链接 torch 绑定会报 librt.so missing：
[ -e /usr/lib/x86_64-linux-gnu/librt.so ] || ln -s /lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so
python3 -m pip wheel . --no-deps -w dist
python3 -m pip install --no-deps --force-reinstall dist/flash_train-*.whl
python3 -c "import flash_train.torch as ft; print(ft.__file__, ft.chunk_kda)"
```

- 最后一行应打印出 `.../site-packages/flash_train/torch/__init__.py` 和 `chunk_kda` 函数。
- 多节点训练时，在一个节点上编译出 `dist/flash_train-*.whl`，其他节点（同一镜像）只需要 `pip install --no-deps` 这个 wheel，不需要上面的编译依赖。
- wheel 里只有 flash-train 自己：`libflash_train.so`（框架加 KDA kernel）、`_flash_train_torch*.so`（PyTorch 绑定）和 Python 代码。torch、DTK 运行时、FLA 都由镜像提供。

#### 2. 修改调用点

**Megatron-LM-k3**：KDA 的 import 在 `megatron/core/ssm/gated_delta_net/kda.py` 文件开头，换成 flash-train 即可，调用处（`self.gated_delta_rule(...)`，`use_qk_l2norm_in_kernel=False`）不用改：

```python
try:
    from flash_train.torch import chunk_kda       # 原来：from fla.ops.kda import chunk_kda
    HAVE_FLA_KDA = HAVE_FLA                        # 卷积、归一化等仍用 FLA 的实现
except ImportError:
    chunk_kda = None
    HAVE_FLA_KDA = False
```

**Megatron-Bridge**：Kimi K3 的 KDA 调用在 `megatron/bridge/models/kimi/kimi_k3_ops.py` 的 `kda()` 里，把 import 换成 flash-train 即可：

```python
def kda(q, k, v, g, beta, a_log, dt_bias, lower_bound, *, cu_seqlens=None):
    from flash_train.torch import chunk_kda       # 原来：from fla.ops.kda import chunk_kda
    output, _ = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,            # q/k/v/g: bf16 [B, T, H, 128]（g 是原始 gate）；beta: fp32 [B, T, H]，已过 sigmoid
        A_log=a_log, dt_bias=dt_bias,             # A_log: [H]，dt_bias: [H, 128] 或 [H*128]，任意浮点类型
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
        lower_bound=lower_bound,                  # K3 为 -5，须在 [-5, 0)
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,                    # packed sequence 时为 cu_seqlens_q（int32 [N+1]，此时 B 必须为 1）
    )
    return output
```

灰度阶段如果要在 FLA 和 flash-train 之间切换，由 Megatron 侧按自己的习惯加开关（配置项或环境变量都可以），flash-train 不规定这个开关。启动命令与 FLA 路线完全相同。

#### 3. 确认走的是 flash-train

- 打印一次路由：在 `kda()` 里加 `print(chunk_kda.__module__)`，应输出 `flash_train.torch.kda`。
- 看 profiler：flash-train 路线下 KDA 的 kernel 名都以 `g2::` 开头（如 `g2::p3_wy_s4b_kernel`、`g2::r4::g2_fwd_prep_a_kernel`），不应再出现 FLA 的 Triton kernel（`chunk_kda_*`、`chunk_gated_delta_rule_*`）。
- 在 gfx938 / gfx936 以外的卡上会直接抛 `RuntimeError`（`FTRAIN_STATUS_UNSUPPORTED`），不会回退到 FLA；需要回退时由 Megatron 侧捕获异常后改用 FLA。
- 反馈问题时请附上：flash-train 版本（`git rev-parse --short HEAD`）、镜像、TP / 序列长度 / 是否 packed / 是否全层重算，以及完整报错。

#### 4. 与 Megatron 的配合

- **TP**：KDA 的 head 按 TP 切分，每卡 H = 总 head 数 / TP（K3 为 96/TP），flash-train 不感知 TP，按传入的 H 计算。
- **packed sequence**：传 `cu_seqlens_q` 即可，各序列互不影响，长度不必是 64 的倍数；`cu_seqlens = [0, T]`（只有一条序列）会自动走更快的 dense kernel。
- **全层重算**（`--recompute-granularity full`）：Megatron 用可重入 checkpoint，第一次前向在 no_grad 下运行，flash-train 不保存任何东西；重算前向按上面的表保存，只活到这一层的反向结束。
- **确定性**：固定顺序归约、不用原子操作，同样的输入多次运行结果逐位相同；与 hip_kda 的独立版本也逐位一致。

#### 5. context parallel（CP > 1）

支持 FLA 的 **chunkwise CP**（`fla.ops.cp.build_cp_context` 造出的 `cp_context`），接口与 FLA 的 `chunk_kda` 完全一致，所以第 2 步换掉 import 之后，CP 的开关、`--context-parallel-size`、调用点全都不用再动。Megatron-LM-k3 的 `kda.py` 在 `cp_size_chunkwise > 1` 时自己构造 `cp_context` 并传进来。`cp_context=None` 走普通路径，所以同一个 wheel CP = 1 和 CP > 1 都能用。

**算法**：每个 rank 先把自己这段序列总结成 `[M | S_loc]`（`S_loc` 是从零状态出发的末状态，`M` 是把值项置零、从单位矩阵出发得到的转移矩阵），`all_gather` 后在 fp32 下把前面各 rank 的总结复合成本 rank 的起始状态 `h0`，再从 `h0` 跑正常递推；反向镜像这一过程（`dh0 = Mᵀ dht + D`），梯度照常穿过 rank 边界。总结复用主调用的前处理，不重复做前处理。首 rank 只算 `S_loc`（它的 `M` 没人用），末 rank 不算总结；中间 rank 的 `S_loc` 和 `M` 在同一趟递推里算出（状态为 `K × (V + K)`）。CP2 没有中间 rank，`all_gather` 只传 `S_loc`。

**gfx938 上的专门处理**（只在 gfx938 上启用；耗时为 nmz5 / BW1102、每 rank 16K token 的实测）：

- 总结递推用编译期确定「只算状态」的实例，不带用不到的输出分支：H = 12–96 时这趟递推快 12–19%（H=24：0.58 → 0.47 ms）；H=8 同时改用窄 slab，0.42 → 0.22 ms。
- H ≤ 8（16 列窄 slab）和 H ≥ 64（64 列宽 slab）的专用前向递推 kernel 加了状态端口，CP 里「从 `h0` 出发的递推」也能用上：H=8 0.60 → 0.42 ms，H=64 1.79 → 1.49 ms。
- 反向总结只需要 `dh0`，不再写用不到的中间结果：H=24 1.01 → 0.78 ms，其他 H 快 7–13%。

**数值**（wheel `1f77b37`，BW1102）：CP2（整条序列 T=4096 / H=24、T=8192 / H=48）、CP3（T=4032 / H=24）、CP4（T=4096 / H=24、T=8192 / H=12、T=4096 / H=8、T=8192 / H=64），输出 `o` 和 `dq/dk/dv/dg/dbeta` 与「单卡一次算完整条序列」**逐位相同**。两个按调用分段求和的标量参数梯度例外：`dt_bias` 的梯度有 2.2e-3–2.4e-3 的相对差异（每段舍入到 bf16，FLA 的 CP 也是同一契约），`A_log` 的梯度 1.5e-7–3.1e-7。`pytest tests` 137 项通过。验证用的每 rank 长度都是 64 token 的整数倍（Megatron 的 32K / 64K 切分即如此），其他长度没有验过。

**限制**（都会显式抛异常，不会静默回退）：

- 每个 rank 一条本地序列：CP 下的 `cu_seqlens` 只能是 `[0, T_local]`，即 **CP 暂不支持 packed 变长序列**；对应地 `micro_batch_size` 需为 1（Megatron 的 `kda.py` 自己也有这个检查）。
- 只支持 chunkwise CP，不支持 headwise CP（`cp_size_headwise > 1`）。
- 要求 `K == V`（K3 是 128 = 128）。
- `cp_context` 与 `initial_state` / `output_final_state=True` 不能同时用。
- `CUDA_DEVICE_MAX_CONNECTIONS=1`（TP/CP 本来就要求）。
- MLA 层在 CP 下跑不起来，报的是 TE 没有可用的 attention backend。这与 KDA 实现无关；CP 的验证用的是 KDA-only 的层配置（`--hybrid-layer-pattern K-KEKE/KE`）。

**算子自身的 CP 开销**（nmz5 / BW1102，每 rank 16K token，rank 0，一次前向+反向，相对不开 CP，交错计时取中位数）：

| 每卡 head 数 | CP2 | CP4 |
|---|---|---|
| H=8 | +26.5% ¹ | 没测 |
| H=12 | +21.1% | +29.1% ² |
| H=24 | +15.6% | +21.6% ² |
| H=48 | +14.6% ¹ | +19.6% ² |

¹ 提交 `df178e7` 时的数据，`1f77b37` 没有重测出可用的结果（H=48 那一轮受其他任务干扰，H=8 没测）。
² 提交 `df178e7` 时的数据；`1f77b37` 只改 CP2 的 `all_gather` 和合并，新旧交替实测 CP4 没有变化。

H=24 时不开 CP 一次前向+反向 9.6 ms，CP2 多约 1.5 ms，CP4 多约 1.9 ms。开销主要是前向、反向各一趟总结递推，它沿 chunk 逐个串行，不随显存带宽变快。同一脚本在 gfx936（`5139c01`）上是 CP2 +22.5% / +17.8% / +13.3%、CP4 +27.2% / +20.4% / +21.2%（H = 12 / 24 / 48）：两种卡的百分比相近，绝对时间 gfx938 更少（H=24 的 CP2：`df178e7` 时 +1.50 ms，gfx936 +2.05 ms）。

**Megatron 端到端：gfx938 上还没有可用的性能数据。** 已有的两次尝试（nmz5，8×BW1102，Megatron-LM-k3，KDA_HEADS 96，GBS 8，MBS 1，adam，KDA-only 层配置，提交 `df178e7`）都和其他任务共用显卡：

- TP4 / CP2 / 32K 跑通了 12 个 iteration（lm loss 10.8126 → 9.583），步时在 2.2–3.2 s 之间波动，不能当作性能数据。
- TP2 / CP4 / 64K：关掉上面三项 gfx938 专门处理的一轮跑通（lm loss 10.7985 → 9.574）；开着的两轮因为显存被其他任务占用而 OOM（每卡 96 GB 只剩约 1.8 GB）。所以默认配置的 CP4 目前只有算子级的验证。
- 同一配置 iteration 1 的 lm loss 在 gfx938 上比 gfx936 高约 0.012（TP4 / CP2 / 32K：10.8126 对 10.8002）。换成 FLA 0.5.2、不开 CP（TP4 / CP1 / 8K）两种卡之间也差 0.0116，flash-train 是 0.0124，所以这个差不是 KDA 算子或 CP 带来的。

#### 参数对照（与 FLA 的差别）

- 只做布局转换，与在打包 THD 张量上直接调用 `kimi_delta_attention` 逐位一致：`[B, T, H, D]` 展平成 `[B*T, H, D]`；`B > 1` 且不传 `cu_seqlens` 时按 B 条长度为 T 的序列处理；传 `cu_seqlens` 时要求 `B == 1`（与 FLA 相同）。
- gate：`use_gate_in_kernel=True` 时要求给出 `A_log`、`dt_bias` 和 `lower_bound`（FLA 的 safe_gate 形式，即 K3 的用法）；`use_gate_in_kernel=False` 时 g 是逐 token log 衰减。`use_beta_sigmoid_in_kernel=True` 时在宿主侧对 beta 做 sigmoid（`allow_neg_eigval` 时乘 2）。
- `disable_recompute` 与 FLA 含义相同，见上一节；不传时用算子自选的策略（FLA 的默认值是 `False`，这里是 `None`）。`chunk_size`、`cu_seqlens_cpu` 是 FLA 的调优参数，这里接受但不起作用。
- `cp_context`（FLA 的 chunkwise CP）支持，见上面第 5 节；与 `initial_state` / `output_final_state=True` 不能同时传。
- 不支持的参数直接抛异常，不会悄悄回退：`return_intermediate_states`、没有 `lower_bound` 的 softplus gate 抛 `NotImplementedError`；未知关键字抛 `TypeError`。

### KDA 调试开关（flash-train 开发用，训练不需要）

这些变量登记在 `include/flash_train/env.h`，只用于 A/B 实验和调试，不改变任何结果的数值。训练脚本里不要设置。

| 变量 | 默认 | 作用 |
|---|---|---|
| `FTRAIN_KDA_SAVE_QKN` | `1` | `0`：前向保存 q/k 而不是 qn/kn，反向重建 qn/kn |
| `FTRAIN_KDA_SAVE_H` | `auto` | `1` 总是保存 h，`0` 从不保存，`auto` 在 H ≤ 64 时保存 |
| `FTRAIN_KDA_KEEP_STATE` | `auto` | 不保存 h 时，`1` 总是交出完整状态，`0` 从不交出，`auto` 只在可重入 checkpoint 的重算前向里交出 |
| `FTRAIN_KDA_DEV` | 不设置 | 设为非空且不为 `0` 时，G2 kernel 才读取从 hip_kda 继承的调参开关（`KDA_G2_*`、`G2_*_STOP`）；不设置时这些开关一律取默认值，训练环境里残留的 hip_kda 变量不会影响 flash-train |

- 前三个只在调用方没有传 `disable_recompute` 时生效，每次前向调用时读取；`FTRAIN_KDA_DEV` 每个进程只在第一次使用时读取一次。

```bash
# 例：A/B 比较"保存 h"与"反向重算"的速度
FTRAIN_KDA_SAVE_H=1 python3 bench.py
FTRAIN_KDA_SAVE_H=0 python3 bench.py
# 例：开发 kernel 时打开 hip_kda 的调参开关
FTRAIN_KDA_DEV=1 KDA_G2_S4_HF=0 python3 bench.py
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
