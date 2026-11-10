# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(
  ftrain_test
  api/common_api_test.cpp
  api/gemm_api_test.cpp
  api/gemm_numerics_test.cpp
  api/moe_api_test.cpp
  api/pattern_api_test.cpp
  api/runtime_api_test.cpp
  cache/cache_test.cpp
  engine/engine_test.cpp
  family/gemm/gemm_test.cpp
  family/moe/family_test.cpp
  family/moe/session_test.cpp
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
endif()

target_link_libraries(ftrain_test PRIVATE flash_train_impl GTest::gtest_main)

add_test(NAME ftrain_test COMMAND ftrain_test)
set_tests_properties(ftrain_test PROPERTIES RESOURCE_LOCK ftrain_hip_devices)

option(FTRAIN_TEST_MOE_MULTIRANK "Register eight-device MoE numerical correctness tests" OFF)
if(FTRAIN_TEST_MOE_MULTIRANK AND NOT FTRAIN_ENABLE_MOE)
  message(FATAL_ERROR "FTRAIN_TEST_MOE_MULTIRANK requires FTRAIN_ENABLE_MOE")
endif()

if(FTRAIN_TEST_MOE_MULTIRANK)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
else()
  find_package(Python3 QUIET COMPONENTS Interpreter)
endif()
if(Python3_Interpreter_FOUND)
  add_test(NAME ftrain_moe_launcher
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/support/run_multirank_test.py")
  set_tests_properties(ftrain_moe_launcher PROPERTIES LABELS "moe;unit" TIMEOUT 30)
endif()

if(FTRAIN_ENABLE_MOE)
  target_sources(ftrain_test PRIVATE api/moe_numerics_test.cpp primitive/hygon/moe/assets_test.cpp)
  target_include_directories(ftrain_test PRIVATE "${PROJECT_SOURCE_DIR}/src/primitive/hygon/moe/include")
  target_compile_definitions(ftrain_test PRIVATE
                            "FTRAIN_MOE_TEST_CODE_OBJECT_DIR=\"${PROJECT_BINARY_DIR}/src/primitive/hygon/moe/code_objects/gfx936\"")
  add_executable(ftrain_moe_environment_test primitive/hygon/moe/default_environment_test.cpp)
  target_include_directories(ftrain_moe_environment_test PRIVATE "${PROJECT_SOURCE_DIR}/src/primitive/hygon/moe")
  target_link_libraries(ftrain_moe_environment_test PRIVATE GTest::gtest_main)
  target_link_options(ftrain_moe_environment_test PRIVATE "LINKER:--wrap=setenv")
  add_test(NAME ftrain_moe_environment COMMAND ftrain_moe_environment_test)
  set_tests_properties(ftrain_moe_environment PROPERTIES LABELS "moe;unit" TIMEOUT 30)
  add_executable(ftrain_moe_perf moe/moe_perf.hip)
  target_link_libraries(ftrain_moe_perf PRIVATE flash_train ${CMAKE_DL_LIBS})
  add_test(NAME ftrain_moe_no_torch
           COMMAND ${CMAKE_COMMAND} -DLIBRARY=$<TARGET_FILE:flash_train>
                   -P "${CMAKE_CURRENT_SOURCE_DIR}/moe/check_no_torch.cmake")
  set_tests_properties(ftrain_moe_no_torch PROPERTIES LABELS "moe;dependency")
  if(FTRAIN_TEST_MOE_MULTIRANK)
    foreach(route IN ITEMS ag_nn_a2a ag_nn_ring ag_nn_ring_chunk ag_nt_a2a ag_nt_a2a_chunk
                           ag_nt_ring_chunk ag_nt_a2a_chunk_32k ag_nt_ring_chunk_32k rs_nn rs_nt)
      if(route MATCHES "^ag_")
        set(kind ag)
      else()
        set(kind rs)
      endif()
      add_test(NAME "ftrain_moe_numerics_${route}"
               COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/support/run_multirank.py"
                       $<TARGET_FILE:ftrain_test> "${kind}" "${route}"
                       "${CMAKE_CURRENT_BINARY_DIR}/moe-numerics/${route}" --world-size 8 --timeout 180 --gtest)
      set_tests_properties("ftrain_moe_numerics_${route}" PROPERTIES
                           LABELS "moe;multirank;numerics" PROCESSORS 8
                           RESOURCE_LOCK ftrain_hip_devices TIMEOUT 200 SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()
