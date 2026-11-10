cmake_minimum_required(VERSION 3.25)
project(ftrain_moe_consumer LANGUAGES C CXX)

find_package(flash_train CONFIG REQUIRED)
add_executable(ftrain_moe_consumer main.c)
target_link_libraries(ftrain_moe_consumer PRIVATE flash_train::flash_train)

option(FTRAIN_CONSUMER_BUILD_SMOKE "Build the installed SDK multi-rank GoogleTest cases" OFF)
if(FTRAIN_CONSUMER_BUILD_SMOKE)
  find_package(GTest CONFIG REQUIRED)
  add_executable(ftrain_test ../../api/moe_numerics_test.cpp)
  target_compile_features(ftrain_test PRIVATE cxx_std_17)
  target_link_libraries(ftrain_test PRIVATE flash_train::flash_train GTest::gtest_main)
endif()
