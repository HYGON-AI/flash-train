# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(
  ftrain_test
  api/common_api_test.cpp
  api/sparse_attn_bwd_api_test.cpp
  api/sparse_attn_bwd_numerics_test.cpp
  cache/cache_test.cpp
  engine/engine_test.cpp
  family/sparse_attn_bwd/sparse_attn_bwd_test.cpp
  error_test.cpp
  plan_test.cpp
  primitive_test.cpp
  primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd_test.cpp
  storage_view_test.cpp
  tensor_test.cpp)

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_compile_options(ftrain_test PRIVATE -Wno-return-type-void-func)
endif()

target_link_libraries(ftrain_test PRIVATE flash_train_impl GTest::gtest_main)

add_test(NAME ftrain_test COMMAND ftrain_test)
