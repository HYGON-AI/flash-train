# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(
  ftrain_test
  api/common_api_test.cpp
  api/gemm_api_test.cpp
  api/gemm_numerics_test.cpp
  api/pattern_api_test.cpp
  api/runtime_api_test.cpp
  cache/cache_test.cpp
  engine/engine_test.cpp
  family/gemm/gemm_test.cpp
  handle/handle_test.cpp
  operation/op_definitions_test.cpp
  operation/op_schema_test.cpp
  logging_test.cpp
  error_test.cpp
  matcher_test.cpp
  ops_args_test.cpp
  pattern_test.cpp
  plan_test.cpp
  primitive_test.cpp
  primitive/hygon/gemm/fp32_gemm_test.cpp
  storage_view_test.cpp
  tensor_test.cpp)

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_compile_options(ftrain_test PRIVATE -Wno-return-type-void-func)
endif()

target_link_libraries(ftrain_test PRIVATE flash_train_impl GTest::gtest_main)

add_test(NAME ftrain_test COMMAND ftrain_test)

# No GPU initialization in this fresh-process probe. Each invocation gets an
# independent read-once environment, including concurrent first-use tests.
add_executable(ftrain_logging_probe logging_probe.cpp)
target_link_libraries(ftrain_logging_probe PRIVATE flash_train_impl)
target_compile_definitions(ftrain_test PRIVATE
  FTRAIN_LOGGING_PROBE_PATH="$<TARGET_FILE:ftrain_logging_probe>")
add_dependencies(ftrain_test ftrain_logging_probe)

add_test(NAME ftrain_engine_logging_debug COMMAND ftrain_test --gtest_filter=OpsEngineLoggingTest.*)
set_tests_properties(ftrain_engine_logging_debug PROPERTIES ENVIRONMENT "FTRAIN_LOG_LEVEL=DEBUG")
