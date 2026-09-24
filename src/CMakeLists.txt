# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

add_library(
  flash_train_impl
  OBJECT api/common_api.cpp
         api/gemm_api.cpp
         api/ops_args_api.cpp
         api/pattern_api.cpp
         api/plan_api.cpp
         ops_args.cpp
         plan.cpp
         constraints.cpp
         error.cpp
         trace.cpp
         storage_view.cpp
         tensor.cpp
         handle.cpp
         cache.cpp
         engine.cpp
         family/gemm/family.cpp
         family/gemm/problem.cpp
         primitive.cpp
         pattern.cpp)
target_include_directories(
  flash_train_impl
  PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include"
         "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(flash_train_impl PUBLIC Threads::Threads)

# The platform-specific primitives: everything under primitive/hygon/
# builds only for the selected platform and carries its dependencies.
if(FTRAIN_PLATFORM_HYGON_HIP)
  target_sources(
    flash_train_impl
    PRIVATE primitive/hygon/gemm/fp32_gemm.cpp
            primitive/hygon/gemm/fp32_gemm.hip)
  target_compile_options(flash_train_impl PRIVATE -Wno-return-type-void-func)
  target_link_libraries(flash_train_impl PUBLIC hip::host)
  target_link_libraries(flash_train_impl PRIVATE ROCTX::roctx)
endif()

if(FTRAIN_BUILD_SHARED_LIBS)
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_BUILDING)
else()
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_STATIC)
endif()

add_library(flash_train ${FTRAIN_LIBRARY_TYPE})
target_sources(flash_train PRIVATE $<TARGET_OBJECTS:flash_train_impl>)
add_library(flash_train::flash_train ALIAS flash_train)
target_link_libraries(flash_train PRIVATE Threads::Threads)

target_sources(
  flash_train
  PUBLIC FILE_SET
  HEADERS
  BASE_DIRS
  "${PROJECT_BINARY_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include"
  FILES
  "${PROJECT_BINARY_DIR}/include/flash_train/control.h"
  "${PROJECT_BINARY_DIR}/include/flash_train/version.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/common.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/env.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/flash_train.h")

if(FTRAIN_BUILD_SHARED_LIBS)
  # A binding build installs the shared library beside the extension module
  # inside the wheel; without VERSION/SOVERSION that copy stays a single
  # unversioned libflash_train.so instead of a symlink chain the wheel
  # format cannot store.
  if(NOT FTRAIN_BUILD_TORCH_EXTENSION)
    set_target_properties(
      flash_train PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION
                                                      ${PROJECT_VERSION_MAJOR})
  endif()
else()
  target_compile_definitions(flash_train PUBLIC FTRAIN_STATIC)
endif()

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_link_libraries(flash_train PUBLIC hip::host)
  target_link_libraries(flash_train PRIVATE ROCTX::roctx)
endif()
