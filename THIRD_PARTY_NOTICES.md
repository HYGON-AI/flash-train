# 第三方组件清单

flash-train 不内置（vendor）任何第三方源码。

运行期第三方组件：

- **PyTorch**：仅 `FTRAIN_BUILD_TORCH_EXTENSION=ON` 时，作为 Python 绑定的构建期与运行期依赖。

以下不作为第三方组件登记：CMake、scikit-build-core、pybind11 等构建与打包工具（不进入发布产物）；Hygon DTK 提供的 HIP 运行时与 ROCTX（Hygon 自有组件）。
