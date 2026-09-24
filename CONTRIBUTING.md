# 参与贡献

欢迎以任何形式参与 flash-train：请求支持新算子、报告问题、提交代码。

## 请求支持新算子

使用[新算子请求 issue 模板](.github/ISSUE_TEMPLATE/new-operator-request.yml)提交，附上算子名称、来源模型/论文与参考实现链接。新算子的支持优先级参考社区需求热度。

## 报告问题

使用 [Bug 报告 issue 模板](.github/ISSUE_TEMPLATE/bug-report.yml)提交，请附最小复现步骤与环境信息（DTK 版本、卡型、库版本）。

## 提交代码

### 开发环境

- HCU 环境：DTK（含 HIP 工具链）与 HCU 卡
- CMake ≥ 3.29、C++17 编译器
- 可选：PyTorch（构建 Python 绑定）

### 构建与测试

```bash
# C 库与测试
cmake -S . -B build \
  -DFTRAIN_PLATFORM_HYGON_HIP=ON \
  -DFTRAIN_BUILD_TESTS=ON \
  -DFTRAIN_HYGON_HIP_ARCHS=gfx938 \
  -Damd_comgr_DIR=/opt/dtk/dcc/comgr/lib64/cmake/amd_comgr \
  -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES=amd_comgr_DIR
cmake --build build
ctest --test-dir build

# flash-train wheel（当前含 PyTorch 绑定；构建参数已写入 pyproject.toml）
python3 -m pip wheel . --no-deps -w dist
```

`amd_comgr_DIR` 请按本机 DTK 安装位置调整。Python 绑定的数值对拍测试位于 `tests/python/`。

### 分支与 PR 流程

- 从 `develop` 拉取特性分支；
- PR 目标分支为 `develop`，所有 CI 检查通过后合入；
- 提交信息遵循 `type(scope): subject` 约定式格式（`feat` / `fix` / `docs` / `refactor` / `ci` / `test` 等）。

### 代码规范

- 新建源码文件顶部加两行声明（C/C++ 用 `//`，CMake/Python 用 `#`；markdown 等不加）：

  ```
  // Copyright (c) 2026 Hygon Information Technology Co., Ltd.
  // SPDX-License-Identifier: MIT
  ```

- 平台名统一使用 HCU；
- 提交前对改动文件运行 clang-format（配置见仓库根 `.clang-format`）；
- 头文件包含分组：C++ 标准库块在前，项目头按依赖层次从低到高排列，文件自己的头放在项目头块最后。

### 测试要求

新算子与新实现需同时提供：

- **白盒测试**：依据实现设计，覆盖确定提供与确定禁止的行为；
- **黑盒测试**：只依据接口注释（行为契约）设计用例，覆盖正常路径与异常条件（含各状态码的触发条件）；
- **数值对拍**：提供 PyTorch 参考实现并接入对拍测试，容差与已知差异文档化。

### 新算子接入指引

添加新算子家族、向现有算子添加实现的完整步骤见 [docs/development.md](docs/development.md)。
