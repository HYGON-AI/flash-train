# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Python bindings: one gated section per framework switch; a new framework
# adds an option in the root plus a section here. The torch module spans two
# translation units because the HCU torch wheel's CUDA-compat headers and
# the HIP runtime headers cannot coexist in one.
# A shared binding build ships libflash_train.so in the package root; the
# framework subpackage __init__ files preload it from there.
if(FTRAIN_BUILD_SHARED_LIBS)
  install(TARGETS flash_train LIBRARY DESTINATION flash_train)
endif()

if(FTRAIN_BUILD_TORCH_BINDINGS)
  find_package(Torch REQUIRED)
  if(TARGET MIOpen)
    get_target_property(FTRAIN_MIOPEN_LINK_LIBRARIES MIOpen INTERFACE_LINK_LIBRARIES)
    foreach(FTRAIN_MIOPEN_LINK_LIBRARY IN LISTS FTRAIN_MIOPEN_LINK_LIBRARIES)
      if(FTRAIN_MIOPEN_LINK_LIBRARY MATCHES "/librt\\.so$"
         AND NOT EXISTS "${FTRAIN_MIOPEN_LINK_LIBRARY}")
        list(REMOVE_ITEM FTRAIN_MIOPEN_LINK_LIBRARIES
             "${FTRAIN_MIOPEN_LINK_LIBRARY}")
        list(APPEND FTRAIN_MIOPEN_LINK_LIBRARIES rt)
      endif()
    endforeach()
    set_property(TARGET MIOpen PROPERTY INTERFACE_LINK_LIBRARIES
                 "${FTRAIN_MIOPEN_LINK_LIBRARIES}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)

  # This HCU TorchConfig defines no torch_python imported target.
  find_library(FTRAIN_TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")

  set(FTRAIN_TORCH_MODULE_EXTRAS "")
  # The Hygon vendored backend relies on symbols remaining in the extension.
  # pybind11's automatic LTO drops those host-side definitions while linking
  # the module, so keep the same NO_EXTRAS behavior as the production backend.
  list(APPEND FTRAIN_TORCH_MODULE_EXTRAS NO_EXTRAS)

  pybind11_add_module(_flash_train_torch ${FTRAIN_TORCH_MODULE_EXTRAS}
                      flash_train/torch/csrc/flash_train_torch.cpp
                      flash_train/torch/csrc/flash_train_torch_capi.cpp
                      flash_train/torch/csrc/hygon_distributed_bind.cpp)
  # The vendored distributed kernels and their checked-in gfx936 code objects
  # are tied to that target.  Keep the general flash_train library multi-arch
  # while making this optional backend gfx936-only.
  set_target_properties(_flash_train_torch PROPERTIES HIP_ARCHITECTURES gfx936)
  target_compile_definitions(_flash_train_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
  target_link_libraries(_flash_train_torch PRIVATE flash_train torch
                                                  "${FTRAIN_TORCH_PYTHON_LIBRARY}")

  # The Hygon distributed backend is compiled into the torch extension.  It
  # must stay outside libflash_train so a wheel can omit the vendored backend
  # without affecting the staged C API.
  #
  # The distributed binding uses flash_train's internal C++ Plan API.  Those
  # symbols are intentionally hidden in libflash_train, so merge the same
  # object files into the extension instead of trying to import them across
  # the shared-library boundary.
  target_sources(_flash_train_torch PRIVATE $<TARGET_OBJECTS:flash_train_impl>)
  target_include_directories(
    _flash_train_torch PRIVATE "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/include")
  set_source_files_properties(
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/ag_gemm/ag_group_gemm.cpp"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/gemm_rs/moe_reduce_rs.cpp"
    PROPERTIES LANGUAGE HIP)
  target_sources(
    _flash_train_torch PRIVATE
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/ag_gemm/ag_group_gemm.cpp"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/gemm_rs/moe_reduce_rs.cpp"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/ths_op/hygon_shm.cc"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/hip_kernels/cudaipc_barrier_all.hip"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/hip_kernels/ag_copy_remote.hip"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/hip_kernels/ag_group_gemm_ck.hip"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/hip_kernels/ag_group_gemm_legacy_compat_ck.hip"
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/vendor/src/hip_kernels/moe_reduce_rs_kernels.hip")
  target_compile_definitions(_flash_train_torch PRIVATE TORCH_CUDA=1 DCU_ASM NDEBUG
                                                         HIP_ENABLE_WARP_SYNC_BUILTINS)
  target_compile_options(_flash_train_torch PRIVATE -O2)
  target_include_directories(_flash_train_torch PRIVATE "${PROJECT_SOURCE_DIR}/src/include")
  target_include_directories(
    _flash_train_torch PRIVATE "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/compat")

  # The vendor session factory must remain dynamically discoverable after the
  # extension is loaded, even though pybind11 defaults the target to hidden C++
  # visibility.
  set_target_properties(
    _flash_train_torch PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
  set_source_files_properties(
    "${PROJECT_SOURCE_DIR}/src/primitive/hygon/distributed/compat/distributed_session.cpp"
    PROPERTIES COMPILE_OPTIONS "-fvisibility=default" CXX_VISIBILITY_PRESET default)

  # scikit-build-core collects the wheel from these install rules.
  install(TARGETS _flash_train_torch LIBRARY DESTINATION flash_train/torch)
  install(DIRECTORY flash_train/torch/hygon_assets/
          DESTINATION flash_train/torch/hygon_assets
          FILES_MATCHING
          PATTERN "*.co")
endif()
