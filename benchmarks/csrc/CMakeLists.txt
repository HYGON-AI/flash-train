# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

add_executable(ftrain_bench_c ftrain_bench.cpp)

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_compile_options(ftrain_bench_c PRIVATE -Wno-return-type-void-func)
endif()

target_link_libraries(ftrain_bench_c PRIVATE flash_train_impl)
