add_library(
  flash_train_impl
  OBJECT api/common_api.cpp
         api/ops_args_api.cpp
         api/pattern_api.cpp
         api/plan_api.cpp
         error.cpp
         gemm.cpp
         ops_engine.cpp
         matcher.cpp
         pattern.cpp
         primitive.cpp
         runtime.cpp
         storage_view.cpp
         tensor.cpp
         trace.cpp)
target_include_directories(
  flash_train_impl
  PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include"
         "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(flash_train_impl PUBLIC Threads::Threads)

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_sources(flash_train_impl PRIVATE kernels/fp32_gemm.hip)
  target_compile_options(flash_train_impl PRIVATE -Wno-return-type-void-func)
endif()

if(FTRAIN_BUILD_SHARED_LIBS)
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_BUILDING)
else()
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_STATIC)
endif()

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_link_libraries(flash_train_impl PUBLIC hip::host)
  target_link_libraries(flash_train_impl PRIVATE ROCTX::roctx)
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
         "${PROJECT_SOURCE_DIR}/include/flash_train/flash_train.h")

if(FTRAIN_BUILD_SHARED_LIBS)
  set_target_properties(
    flash_train PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION
                                                      ${PROJECT_VERSION_MAJOR})
else()
  target_compile_definitions(flash_train PUBLIC FTRAIN_STATIC)
endif()

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_link_libraries(flash_train PUBLIC hip::host)
  target_link_libraries(flash_train PRIVATE ROCTX::roctx)
endif()
