# Flash Train

Flash Train 是面向大模型和通用模型训练的算子库，统一支持单算子、融合算子、低阶算子和高阶算子。

算子库支持 FP32、BF16、FP16 直接计算及融合计算，支持高精度到低精度的量化、低精度到高精度的反量化，以及 GroupedGEMM 融合和通算融合。上层使用者包括 PyTorch、JAX、Megatron 等框架。

公共 C 接口位于 `include/flash_train`。

## 目录

- `include/flash_train`：公共 C 接口。
- `src`：算子库实现。
- `src/include/flash_train`：内部 C++ 接口。
- `src/api`：公共 C 接口实现。
- `cmake`：构建与安装配置。
