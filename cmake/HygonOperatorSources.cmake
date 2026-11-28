# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Architecture manifests register only the sources they actually provide.
# The architecture list itself is owned by the top-level CMake configuration.
function(ftrain_add_hygon_operator_target name arch)
  if(NOT arch IN_LIST FTRAIN_HYGON_HIP_ARCHS)
    message(FATAL_ERROR "${name}: ${arch} is not requested by FTRAIN_HYGON_HIP_ARCHS")
  endif()
  if(NOT ARGN)
    message(FATAL_ERROR "${name}: no HIP sources")
  endif()
  add_library(${name} OBJECT ${ARGN})
  set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES "${arch}"
                                           POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${name} PRIVATE -Wno-return-type)
  target_link_libraries(${name} PRIVATE hip::host)
  set_property(GLOBAL APPEND PROPERTY FTRAIN_HYGON_OPERATOR_TARGETS ${name})
endfunction()
