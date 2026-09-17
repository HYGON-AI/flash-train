add_executable(
  ftrain_test
  common_api_test.cpp
  gemm_api_test.cpp
  pattern_api_test.cpp
  runtime_api_test.cpp
  ops_args_test.cpp
  error_test.cpp
  storage_view_test.cpp
  tensor_test.cpp
  ops_engine_test.cpp
  gemm_test.cpp
  matcher_test.cpp
  primitive_test.cpp
  op_definitions_test.cpp
  op_schema_test.cpp
  pattern_test.cpp)

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_compile_options(ftrain_test PRIVATE -Wno-return-type-void-func)
endif()

target_link_libraries(ftrain_test PRIVATE flash_train_impl GTest::gtest_main)

add_test(NAME ftrain_test COMMAND ftrain_test)
