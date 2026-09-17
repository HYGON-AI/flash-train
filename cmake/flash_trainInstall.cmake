# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

include(CMakePackageConfigHelpers)
install(
  TARGETS flash_train
  EXPORT flash_trainTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
          COMPONENT Runtime
          NAMELINK_COMPONENT Development
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
          COMPONENT Development
          FILE_SET HEADERS
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
          COMPONENT Development)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/flash_trainConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/cmake/flash_trainConfig.cmake"
  INSTALL_DESTINATION "${FTRAIN_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/flash_trainConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY "SameMajorVersion")

install(
  EXPORT flash_trainTargets
  FILE flash_trainTargets.cmake
  NAMESPACE flash_train::
  DESTINATION "${FTRAIN_INSTALL_CMAKEDIR}"
  COMPONENT Development)

install(
  FILES "${PROJECT_BINARY_DIR}/cmake/flash_trainConfig.cmake"
        "${PROJECT_BINARY_DIR}/cmake/flash_trainConfigVersion.cmake"
  DESTINATION "${FTRAIN_INSTALL_CMAKEDIR}"
  COMPONENT Development)

if(FTRAIN_PLATFORM_HYGON_HIP AND NOT FTRAIN_BUILD_SHARED_LIBS)
  install(
    FILES "${PROJECT_SOURCE_DIR}/cmake/FindROCTX.cmake"
    DESTINATION "${FTRAIN_INSTALL_CMAKEDIR}"
    COMPONENT Development)
endif()
