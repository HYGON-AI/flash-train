# flash-train 开发指南

两类开发任务：给库**添加一个全新算子**，或**向现有算子添加一个实现**。前者是全链路工作，后者只触碰家族目录内两处。

## 1. 添加全新算子

以新增算子 `Foo` 为例，自底向上六步。参照实现：`sparse_attn_bwd` 各层的对应文件。

### 第 1 步：family 层——家族资料

在 `family/foo/` 下建三个头文件：

- **`problem.hpp` + `src/family/foo/problem.cpp`** —— `FooProblem` 清单类：
  - 校验构造器：把跨字段结构校验（秩、形状一致、字节溢出、内存放置）全部吸收，让到达 Primitive 的问题天然合法；
  - 逐端口 getter：算子开发者逐函数过一遍即知需要支持什么；
  - `getProblemKey()`：把一切可能改变选择的属性编码为整数 token（注释里有缓存正确性契约），**禁止编码裸地址**。
  - 只属于这个算子的调用属性（如计算精度）放在同文件的属性类里，作为 Problem 构造器的最后一个参数。
- **`finder.hpp`** —— 选择策略：`getName()` + `findCandidates(FooProblem, Constraints) → 名字数组`（按偏好序）。初期可直接抄占位实现。
- **`family.hpp` + `src/family/foo/family.cpp`** —— `FooFamily` 策略：`using Problem`、`makeRecords`（注册实现）；同文件给出引擎别名 `using FooEngine = OpEngine<FooFamily, Foo…Finder>;` 和进程级实例的访问函数 `const FooEngine& fooEngine();`（函数内静态对象，首次调用时构造）。`makeRecords` 声明在头文件、定义在 cpp——实现清单和各 Primitive 头不进 hpp，家族壮大后头文件不膨胀。

### 第 2 步：primitive 层——至少一个实现

`primitive/hygon/foo/` 下实现 `Primitive<FooProblem>`（平台子树，平台相关代码全部收在此处）：

- `isApplicable`：只回答"支不支持这些值"（结构合法性已由 Problem 构造器保证）；
- `configure`：把执行所需的一切按值存入（Problem 随后可销毁）；
- `executeImpl` + 内核：在传入流上异步提交，**不做同步**；
- `getName`/`clone`/`getRequiredWorkspaceBytes` 按基类契约实现。

### 第 3 步：公共 C API——这个算子的函数族

句柄类型和分阶段函数声明在 `include/flash_train/common.h`，便利函数声明在 `include/flash_train/flash_train.h`，实现放在 `src/api/foo_api.cpp`：

- 句柄类型 `typedef struct FTrainFooPlanStruct* FTrainFooPlan;`；句柄的内部结构（`FTrainFooPlanStruct`，包一个 `ftrain::Plan`）定义在 `src/include/flash_train/api.hpp`；
- 分阶段函数，全部以算子名开头：`ftrainFooCreate`（参数是算子的全部端口视图加属性，返回句柄）、`ftrainFooDestroy`、`ftrainFooGetNumPrimitives`、`ftrainFooGetPrimitiveName`、`ftrainFooGetPrimitiveRequiredWorkspaceBytes`、`ftrainFooExecute`；
- 便利函数 `ftrainFoo`：参数与 `Create` 相同再加 workspace 与 stream，内部建 Plan 并执行下标 0。

`Create` 的内部只有三件事：校验属性枚举 → 用 `makeTensor` 把 `FTrainStorageView` 转成 `Tensor` 并构造 `FooProblem` → `fooEngine().createPlan(problem)`。其余五个函数都是转发到句柄里 `Plan` 的同名方法。公共头里的注释就是调用方契约：只读注释即可写出黑盒测试。

### 第 4 步：测试与构建

- **白盒直写**（参照 `tests/primitive/hygon/sparse_attn_bwd/` 与 `tests/family/sparse_attn_bwd/`）：Problem 校验用例、isApplicable 清单逐项、`makeRecords()` 的名字与 clone 唯一性，不套额外框架；
- **黑盒行为**：算子 fixture 继承 `tests/support/staged_api.hpp` 的 `StagedApiTest`（流生命周期、显存登记、视图构造、状态断言由基类持有），在 fixture 里包装这个算子的 `Create`，参照 `tests/api/sparse_attn_bwd_staged_fixture.hpp`；黑盒测试只依据公共头里的注释契约，覆盖各状态码的触发条件；
- **数值一致性（C++）**：`tests/support/numerics.hpp` 提供 `DeviceBuffer` 与 `expectNear`（混合绝对/相对容差，失败信息带最大偏差与位置）；算子侧交付三样——host 参考实现、形状网格（含非 2 的幂边界）、容差取值及依据（写进用例注释），参照 `tests/api/sparse_attn_bwd_numerics_test.cpp`；
- **数值一致性（Python 绑定）**：`tests/python/` 参数化形状网格对拍 torch 参考实现，参照 `torch_sparse_attn_bwd_test.py`；
- **PyTorch 绑定**：在 `python/flash_train/torch/csrc/flash_train_torch_capi.cpp` 加桥接函数（把张量指针与维度转成 `FTrainStorageView` 并调用上面的 C 函数），在 `flash_train_torch.cpp` 加 pybind 封装并在模块里注册；
- `src/CMakeLists.txt` 登记库源文件、`tests/CMakeLists.txt` 登记测试源文件，并在 HCU 环境完成构建与测试验证（构建与测试步骤见 [CONTRIBUTING.md](https://github.com/HYGON-AI/flash-train/blob/develop/CONTRIBUTING.md)）。

### 第 5 步：基准接入

新算子按两侧接入（口径与方法论见 [benchmarks/README.md](https://github.com/HYGON-AI/flash-train/blob/develop/benchmarks/README.md)）：

- **Python 侧四件套**：`benchmarks/ftrain_bench/ops/<op>.py` 声明 `make_args`（固定种子生成张量）、`baseline`（PyTorch 组合实现）、`ftrain`（被测调用——被测库在调用点才导入，编排与渲染路径不依赖 wheel 已安装）与 `suites`（标准形状集），并在 `ops/__init__.py` 导入注册；
- **C harness 侧**：`benchmarks/csrc/ops/<op>.cpp` 实现 `OpCase`/`OpSession`——`runConvenience` 调便利函数，`createPlan` 返回一个把该算子句柄接到 `PlanSession` 接口上的对象（参照 `sparse_attn_bwd.cpp`），在 `benchmarks/CMakeLists.txt` 登记源文件。三档口径（C 便利 / Plan 复用 / 逐 Primitive）由通用骨架自动获得；
- 标准形状集一经发布即固定，改动视为口径变更并需在结果元数据中可见。

### 第 6 步：文档

- `docs/usage.md`：三级接入各加一节；
- `README.md` 与 `docs/index.md` 的算子目录加一行；
- 结果入库后，把 `benchmarks/README.md` 与 `mkdocs.yml` 导航里的基准页面补上。

## 2. 向现有家族添加实现

以给 sparse_attn_bwd 增加 `Fp16SparseAttnBwd` 为例，只碰两处：

1. **实现**：`primitive/hygon/sparse_attn_bwd/fp16_sparse_attn_bwd.{hpp,cpp,hip}`，实现 `Primitive<SparseAttnBwdProblem>` 全套接口；
2. **注册**：`src/family/sparse_attn_bwd/family.cpp` 包含新头 + `makeRecords()` 加一项。

随后的行为自动获得：名字唯一性校验（重名注册即拒）、按名环境变量过滤、Finder 候选与注册序兜底、选择缓存、`ftrainSparseAttnBwdGetNumPrimitives` 遍历可见。基准同样无需改动——通用骨架按下标枚举 Plan 内全部实现，新 primitive 自动进入实现矩阵。

测试同上：`tests/primitive/hygon/sparse_attn_bwd/<impl>_test.cpp` 白盒 + API 黑盒。

## 3. 通用约束（两类任务都适用）

- **命名即身份**：`getName()` 在引擎内唯一，进入 `FTRAIN_ENABLED/DISABLED_PRIMITIVES` 语义；生命周期须覆盖对象存活期。
- **isApplicable 与缓存契约**：缓存命中会跳过 isApplicable，故 Problem key 必须覆盖一切能改变适用性的属性；不要在 isApplicable 里读裸地址之外的未编码信息。
- **clone 独立性**：`clone()` 产物被 Plan 持有并 configure，成员必须是值或非拥有指针。
- **头文件包含**遵循仓库规则：标准库块在前、项目块按依赖层次排、own header 收尾、最小显式清单。
