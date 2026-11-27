# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_library(
  flash_train_impl
  OBJECT api/common_api.cpp
         api/gdn_api.cpp
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
         family/gdn/family.cpp
         family/gdn/problem.cpp
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
  include("${PROJECT_SOURCE_DIR}/cmake/HygonOperatorSources.cmake")
  foreach(FTRAIN_OPERATOR_ARCH IN LISTS FTRAIN_HYGON_HIP_ARCHS)
    foreach(ftrain_operator IN ITEMS gdn gdn2)
      set(ftrain_operator_sources "primitive/hygon/${ftrain_operator}/${FTRAIN_OPERATOR_ARCH}/CMakeLists.txt")
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ftrain_operator_sources}")
        add_subdirectory("primitive/hygon/${ftrain_operator}/${FTRAIN_OPERATOR_ARCH}")
      else()
        message(STATUS "${ftrain_operator}: no ${FTRAIN_OPERATOR_ARCH} device sources; runtime unsupported")
      endif()
    endforeach()
  endforeach()
  target_sources(
    flash_train_impl
    PRIVATE primitive/hygon/gemm/fp32_gemm.cpp
            primitive/hygon/gemm/fp32_gemm.hip)
  # Per-architecture generic availability macros: for every architecture
  # whose complete device target set was built, FTRAIN_HAS_<ARCH_UPPER> is
  # defined (FTRAIN_HAS_GFX936, FTRAIN_HAS_GFX938, ...).  The macros gate
  # both the registered records (family.cpp, compiled here) and the public
  # Finder header, so they are PUBLIC: test and other consumers of
  # finder.hpp must see the same values.  Operator code never invents its
  # own arch macros; it gates arch-specific host code on these generic ones,
  # and the family registration below keeps the registered records in sync
  # with the linked device code objects.  With the default list both macros
  # are defined and selection is purely run time (Constraints::getArch());
  # a single-architecture subset list keeps only that architecture's
  # records and the build stays linkable.
  if(TARGET ftrain_gdn_gfx936 AND TARGET ftrain_gdn_four_gfx936 AND
     TARGET ftrain_gdn_bwd_gfx936 AND TARGET ftrain_gdn2_gfx936 AND
     TARGET ftrain_gdn2_bwd_gfx936)
    target_sources(
      flash_train_impl PRIVATE
      primitive/hygon/gdn/fused_forward.cpp
      primitive/hygon/gdn/four_forward.cpp
      primitive/hygon/gdn/chunk_backward.cpp
      primitive/hygon/gdn2/chunk_forward.cpp
      primitive/hygon/gdn2/chunk_backward.cpp)
    target_compile_definitions(flash_train_impl PUBLIC FTRAIN_HAS_GFX936=1)
  endif()
  if(TARGET ftrain_gdn_gfx938 AND TARGET ftrain_gdn_bwd_gfx938 AND
     TARGET ftrain_gdn2_gfx938 AND TARGET ftrain_gdn2_bwd_gfx938)
    target_sources(
      flash_train_impl PRIVATE
      primitive/hygon/gdn/fused_forward_gfx938.cpp
      primitive/hygon/gdn/chunk_backward_gfx938.cpp
      primitive/hygon/gdn2/chunk_forward_gfx938.cpp
      primitive/hygon/gdn2/chunk_backward_gfx938.cpp)
    target_compile_definitions(flash_train_impl PUBLIC FTRAIN_HAS_GFX938=1)
  endif()
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
if(FTRAIN_PLATFORM_HYGON_HIP)
  get_property(ftrain_hygon_operator_targets GLOBAL PROPERTY FTRAIN_HYGON_OPERATOR_TARGETS)
  foreach(ftrain_target IN LISTS ftrain_hygon_operator_targets)
    target_sources(flash_train PRIVATE $<TARGET_OBJECTS:${ftrain_target}>)
  endforeach()
endif()
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
  # A binding build ships the library inside the wheel as one unversioned
  # file next to the extension module.
  if(NOT FTRAIN_BUILD_WHEEL)
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
