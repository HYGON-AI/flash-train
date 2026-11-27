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
  family/gdn/finder_test.cpp
  family/gdn/gdn_problem_test.cpp
  family/gdn/constraints_arch_test.cpp
  family/gdn2/gdn2_problem_test.cpp
  handle/handle_test.cpp
  operation/op_definitions_test.cpp
  operation/op_schema_test.cpp
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
  get_property(ftrain_hygon_operator_targets GLOBAL PROPERTY FTRAIN_HYGON_OPERATOR_TARGETS)
  foreach(ftrain_target IN LISTS ftrain_hygon_operator_targets)
    target_link_libraries(ftrain_test PRIVATE ${ftrain_target})
  endforeach()
  if(TARGET ftrain_gdn_gfx936 AND TARGET ftrain_gdn_four_gfx936 AND
     TARGET ftrain_gdn_bwd_gfx936)
    target_sources(ftrain_test PRIVATE
                   api/gdn_forward_numerics_test.cpp
                   api/gdn_backward_numerics_test.cpp
                   family/gdn/fused_forward_test.cpp)
  endif()
  if(TARGET ftrain_gdn2_gfx936 AND TARGET ftrain_gdn2_bwd_gfx936)
    target_sources(ftrain_test PRIVATE
                   api/gdn2_forward_numerics_test.cpp
                   api/gdn2_backward_numerics_test.cpp
                   family/gdn2/chunk_forward_test.cpp)
  endif()
  if(TARGET ftrain_gdn_gfx936 AND TARGET ftrain_gdn_four_gfx936 AND
     TARGET ftrain_gdn_bwd_gfx936 AND TARGET ftrain_gdn2_gfx936 AND
     TARGET ftrain_gdn2_bwd_gfx936)
    target_sources(ftrain_test PRIVATE api/gdn_api_test.cpp)
  endif()
  if(TARGET ftrain_gdn_gfx938 AND TARGET ftrain_gdn_bwd_gfx938)
    target_sources(ftrain_test PRIVATE
                   api/gdn_forward_numerics_gfx938_test.cpp
                   api/gdn_backward_numerics_gfx938_test.cpp
                   family/gdn/fused_forward_gfx938_test.cpp)
  endif()
  if(TARGET ftrain_gdn2_gfx938 AND TARGET ftrain_gdn2_bwd_gfx938)
    target_sources(ftrain_test PRIVATE
                   api/gdn2_forward_numerics_gfx938_test.cpp
                   api/gdn2_backward_numerics_gfx938_test.cpp
                   family/gdn2/chunk_forward_gfx938_test.cpp)
  endif()
endif()

target_link_libraries(ftrain_test PRIVATE flash_train_impl GTest::gtest_main)

add_test(NAME ftrain_test COMMAND ftrain_test)
