# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# 新算子接入：在 ops/ 下新增实现（参照 ops/gemm.cpp）并在此登记源文件。
add_executable(ftrain_bench_c csrc/ftrain_bench.cpp csrc/ops/gemm.cpp)

target_include_directories(ftrain_bench_c PRIVATE csrc)
target_link_libraries(ftrain_bench_c PRIVATE flash_train_impl)
