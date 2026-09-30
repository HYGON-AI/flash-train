# flash-train 开发指南

两类开发任务：给库**添加一个全新 Pattern**（新算子家族），或**向现有 Pattern 添加一个实现**。前者是全链路工作，后者只触碰家族目录内两处。

## 1. 添加全新 Pattern（新算子家族）

以新增算子 `Foo` 为例，自底向上五步。前置判断：若 `OperationKind` 尚无 `kFoo`，从第 1 步开始；若只是已有算子种类的新拓扑（如另一条 Gemm 链），跳到第 2 步。

### 第 1 步：operation 层——声明算子的拓扑类型

| 文件 | 动作 |
|---|---|
| `src/include/flash_train/operation/base.hpp` | `OperationKind` 加 `kFoo` |
| `src/include/flash_train/operation/foo.hpp`（新建） | `FooAttributes` 属性类；`OperationTraits<kFoo>` 特化：`Id`/`Type`/`createPatternOperationNode`（端口接线校验）/`createPatternOperationId` |
| `src/api/pattern_api.cpp` / `ops_args_api.cpp` + `include/flash_train/common.h` | `FTrainFooOpId` 类型、`ftrainPatternAddFoo`（用户拼 Pattern 时接线）、`ftrainArgsSetFoo`（填属性） |

对照样例：`operation/gemm.hpp` 与 `ftrainPatternAddGemm`/`ftrainArgsSetGemm`。

### 第 2 步：family 层——家族资料

在 `family/foo/` 下建三个头文件（Gemm 家族是完整参照）：

- **`problem.hpp` + `src/family/foo/problem.cpp`** —— `FooProblem` 清单类：
  - 校验构造器：把跨字段结构校验（秩、k 一致、形状、字节溢出、内存放置）全部吸收，让到达 Primitive 的问题天然合法；
  - 逐端口 getter：算子开发者逐函数过一遍即知需要支持什么；
  - `getProblemKey()`：把一切可能改变选择的属性编码为整数 token（注释里有缓存正确性契约），**禁止编码裸地址**。
- **`finder.hpp`** —— 选择策略：`getName()` + `findCandidates(FooProblem, Constraints) → 名字数组`（按偏好序）。初期可直接抄占位实现。
- **`family.hpp` + `src/family/foo/family.cpp`** —— `FooFamily` 策略：`Roles` + `makeRoles()`（用 builder 返回的角色 ID 绑定端口，杜绝手工编号）、`makePattern`、`makeProblem`（用保存的 ID 取端口）；`makeRecords`（注册实现）声明在头文件、定义在 cpp——实现清单和各 Primitive 头不进 hpp，家族壮大后头文件不膨胀。

### 第 3 步：primitive 层——至少一个实现

`primitive/hygon/foo/` 下实现 `Primitive<FooProblem>`（平台子树，平台相关代码全部收在此处）：

- `isApplicable`：只回答"支不支持这些值"（结构合法性已由 Problem 构造器保证）；
- `configure`：把执行所需的一切按值存入（Problem 随后可销毁）；
- `executeImpl` + 内核：在传入流上异步提交，**不做同步**；
- `getName`/`clone`/`getRequiredWorkspaceBytes` 按基类契约实现。

### 第 4 步：注册

`src/engine.cpp` 的 `makeBuiltinOpsEngines()` 里直接构造 `OpsEngine<FooFamily, ...Finder>`——全库唯一注册点。

### 第 5 步：测试与构建

- **白盒直写**（参照 `tests/family/gemm/` 与 `tests/primitive/hygon/gemm/`）：Problem 校验用例、isApplicable 清单逐项，不套额外框架；
- **黑盒行为**：算子 fixture 继承 `tests/support/staged_api.hpp` 的 `StagedApiTest`（流/句柄生命周期、显存登记、视图构造、状态断言由基类持有），在 SetUp 中装配自己的 Pattern 与角色绑定，参照 `tests/api/gemm_staged_fixture.hpp`；
- **数值一致性（C++）**：`tests/support/numerics.hpp` 提供 `DeviceBuffer` 与 `expectNear`（混合绝对/相对容差，失败信息带最大偏差与位置）；算子侧交付三样——host 参考实现、形状网格（含非 2 的幂边界）、容差取值及依据（写进用例注释），参照 `tests/api/gemm_numerics_test.cpp`；
- **数值一致性（Python 绑定）**：`tests/python/` 参数化形状网格对拍 torch 参考实现，参照 `torch_gemm_test.py`；
- `src/CMakeLists.txt` 登记库源文件、`tests/CMakeLists.txt` 登记测试源文件，并在 HCU 环境完成构建与测试验证（构建与测试步骤见 [CONTRIBUTING.md](https://github.com/HYGON-AI/flash-train/blob/develop/CONTRIBUTING.md)）。

### 第 6 步：基准接入

新算子按两侧接入（口径与方法论见 [benchmarks/README.md](https://github.com/HYGON-AI/flash-train/blob/develop/benchmarks/README.md)）：

- **Python 侧四件套**：`benchmarks/ftrain_bench/ops/<op>.py` 声明 `make_args`（固定种子生成张量）、`baseline`（PyTorch 组合实现）、`ftrain`（被测调用——被测库在调用点才导入，编排与渲染路径不依赖 wheel 已安装）与 `suites`（标准形状集），并在 `ops/__init__.py` 导入注册；
- **C harness 侧三钩子**：`benchmarks/csrc/ops/<op>.cpp` 实现 `OpCase`/`OpSession`（参照 `gemm.cpp`：由形状构造端口视图、便利调用、Pattern/Args 装配），在 `benchmarks/CMakeLists.txt` 登记源文件。三档口径（C 便利 / Plan 复用 / 逐 Primitive）由通用骨架自动获得；
- 标准形状集一经发布即固定，改动视为口径变更并需在结果元数据中可见。

## 2. 向现有 Pattern 添加实现

以给 Gemm 增加 `Fp16Gemm` 为例，只碰两处：

1. **实现**：`primitive/hygon/gemm/fp16_gemm.{hpp,cpp,hip}`，实现 `Primitive<GemmProblem>` 全套接口；
2. **注册**：`src/family/gemm/family.cpp` 包含新头 + `makeRecords()` 加一项。

随后的行为自动获得：名字唯一性校验（重名注册即拒）、按名环境变量过滤、Finder 候选与注册序兜底、选择缓存、API 层 `ftrainPlanGetNumPrimitives` 遍历可见。基准同样无需改动——L4 按下标枚举 Plan 内全部实现，新 primitive 自动进入实现矩阵。

测试同上：`tests/primitive/hygon/gemm/fp16_gemm_test.cpp` 白盒 + API 黑盒。

## 3. 通用约束（两类任务都适用）

- **命名即身份**：`getName()` 在引擎内唯一，进入 `FTRAIN_ENABLED/DISABLED_PRIMITIVES` 语义；生命周期须覆盖对象存活期。
- **isApplicable 与缓存契约**：缓存命中会跳过 isApplicable，故 Problem key 必须覆盖一切能改变适用性的属性；不要在 isApplicable 里读裸地址之外的未编码信息。
- **clone 独立性**：`clone()` 产物被 Plan 持有并 configure，成员必须是值或非拥有指针。
- **头文件包含**遵循仓库规则：标准库块在前、项目块按依赖层次排、own header 收尾、最小显式清单。
