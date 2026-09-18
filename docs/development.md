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

`primitive/foo/` 下实现 `Primitive<FooProblem>`：

- `isApplicable`：只回答"支不支持这些值"（结构合法性已由 Problem 构造器保证）；
- `configure`：把执行所需的一切按值存入（Problem 随后可销毁）；
- `executeImpl` + 内核：在传入流上异步提交，**不做同步**；
- `getName`/`clone`/`getRequiredWorkspaceBytes` 按基类契约实现。

### 第 4 步：注册

`src/engine.cpp` 的 `makeBuiltinOpsEngines()` 里直接构造 `OpsEngine<FooFamily, ...Finder>`——全库唯一注册点。

### 第 5 步：测试与构建

- 白盒：`tests/family/foo/`（Problem 校验）、`tests/primitive/foo/`（isApplicable 清单逐项）；
- 黑盒：`tests/api/` 端到端（C API 全流程 + 数值对拍）；
- `src/CMakeLists.txt` 登记源文件；远程 HCU 节点构建验证（见 AGENTS.md）。

## 2. 向现有 Pattern 添加实现

以给 Gemm 增加 `Fp16Gemm` 为例，只碰两处：

1. **实现**：`primitive/gemm/fp16_gemm.{hpp,cpp,hip}`，实现 `Primitive<GemmProblem>` 全套接口；
2. **注册**：`src/family/gemm/family.cpp` 包含新头 + `makeRecords()` 加一项。

随后的行为自动获得：名字唯一性校验（重名注册即拒）、按名环境变量过滤、Finder 候选与注册序兜底、选择缓存、API 层 `ftrainPlanGetNumPrimitives` 遍历可见。

测试同上：`tests/primitive/gemm/fp16_gemm_test.cpp` 白盒 + API 黑盒。

## 3. 通用约束（两类任务都适用）

- **命名即身份**：`getName()` 在引擎内唯一，进入 `FTRAIN_ENABLED/DISABLED_PRIMITIVES` 语义；生命周期须覆盖对象存活期。
- **isApplicable 与缓存契约**：缓存命中会跳过 isApplicable，故 Problem key 必须覆盖一切能改变适用性的属性；不要在 isApplicable 里读裸地址之外的未编码信息。
- **clone 独立性**：`clone()` 产物被 Plan 持有并 configure，成员必须是值或非拥有指针。
- **头文件包含**遵循仓库规则（见 AGENTS.md）：标准库块在前、项目块按依赖层次排、own header 收尾、最小显式清单。
