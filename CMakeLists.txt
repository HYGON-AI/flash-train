# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.29)

# Auto project settings
set(FTRAIN_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${FTRAIN_VERSION_FILE}")
file(STRINGS "${FTRAIN_VERSION_FILE}" FTRAIN_PROJECT_VERSION LIMIT_COUNT 1)
project(
  flash_train
  VERSION ${FTRAIN_PROJECT_VERSION}
  LANGUAGES CXX)

# Developer configurations
option(FTRAIN_BUILD_SHARED_LIBS "Build flash_train as a shared library" ON)
option(FTRAIN_INSTALL "Generate installation rules" ${PROJECT_IS_TOP_LEVEL})
option(FTRAIN_BUILD_TESTS "Build flash_train tests" OFF)
option(FTRAIN_BUILD_TORCH_EXTENSION "Build the ftrain_torch PyTorch binding" OFF)
option(FTRAIN_PLATFORM_HYGON_HIP "flash_train for hygon hip platform" OFF)
set(FTRAIN_HYGON_HIP_ARCHS
    "gfx936;gfx938"
    CACHE STRING
          "Hygon hip architectures; used only when FTRAIN_PLATFORM_HYGON_HIP=ON"
)
mark_as_advanced(FTRAIN_INSTALL)
mark_as_advanced(FTRAIN_HYGON_HIP_ARCHS)

# Check developer configurations
if(NOT FTRAIN_PLATFORM_HYGON_HIP)
  message(FATAL_ERROR "Must enable one of options 'FTRAIN_PLATFORM_HYGON_HIP'")
endif()

# Auto configure according to developer configurations
include(GNUInstallDirs)
if(FTRAIN_BUILD_SHARED_LIBS)
  set(FTRAIN_LIBRARY_TYPE SHARED)
else()
  set(FTRAIN_LIBRARY_TYPE STATIC)
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(FTRAIN_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/flash_train")
# set(CMAKE_SKIP_RPATH True)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(FTRAIN_PLATFORM_HYGON_HIP)
  set(CMAKE_HIP_ARCHITECTURES "${FTRAIN_HYGON_HIP_ARCHS}")
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
  set(CMAKE_HIP_VISIBILITY_PRESET hidden)
  find_package(hip CONFIG REQUIRED)
  find_package(ROCTX MODULE REQUIRED)
endif()

find_package(Threads REQUIRED)

configure_file("${PROJECT_SOURCE_DIR}/include/flash_train/control.h.in"
               "${PROJECT_BINARY_DIR}/include/flash_train/control.h" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/include/flash_train/version.h.in"
               "${PROJECT_BINARY_DIR}/include/flash_train/version.h" @ONLY)

add_subdirectory(src)

if(FTRAIN_BUILD_TORCH_EXTENSION)
  add_subdirectory(python)
endif()

if(FTRAIN_BUILD_TESTS)
  enable_testing()

  if(NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
      URL_HASH
        SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(googletest)

    if(FTRAIN_PLATFORM_HYGON_HIP)
      foreach(ftrain_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
        target_compile_options(${ftrain_gtest_target}
                               PRIVATE -Wno-return-type-void-func)
      endforeach()
    endif()
  endif()
  add_subdirectory(tests)
endif()

# Auto install according to developer configurations. Binding builds keep
# the C SDK rules off: scikit-build-core collects whatever install rules
# exist into the wheel, so only the python/ rules may fire there.
if(FTRAIN_INSTALL AND NOT FTRAIN_BUILD_TORCH_EXTENSION)
  include("${PROJECT_SOURCE_DIR}/cmake/flash_trainInstall.cmake")
endif()
