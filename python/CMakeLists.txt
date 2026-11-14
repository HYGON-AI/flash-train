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
  find_package(pybind11 CONFIG REQUIRED)

  # This HCU TorchConfig defines no torch_python imported target.
  find_library(FTRAIN_TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")

  set(FTRAIN_TORCH_MODULE_EXTRAS "")
  if(NOT FTRAIN_BUILD_SHARED_LIBS)
    # pybind11's default LTO breaks GNU ld's static-archive member extraction
    # on this toolchain, silently dropping libflash_train.
    list(APPEND FTRAIN_TORCH_MODULE_EXTRAS NO_EXTRAS)
  endif()

  pybind11_add_module(_flash_train_torch ${FTRAIN_TORCH_MODULE_EXTRAS}
                      flash_train/torch/csrc/flash_train_torch.cpp
                      flash_train/torch/csrc/flash_train_torch_capi.cpp)
  target_compile_definitions(_flash_train_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
  # The HCU torch exports glog, which drags libunwind in as a direct dependency of this module, ahead
  # of libgcc_s. Under a Python that does not load libgcc_s itself (e.g. the /usr/local Python 3.12 of
  # the dtk2610 Megatron images) the module's _Unwind_Resume then binds to libunwind while libstdc++'s
  # personality routine uses libgcc, and every C++ exception (TORCH_CHECK) aborts the process instead
  # of raising. Naming libgcc_s first keeps the whole unwind on libgcc.
  target_link_options(_flash_train_torch PRIVATE "LINKER:--push-state,--no-as-needed" "-lgcc_s"
                      "LINKER:--pop-state")
  target_link_libraries(_flash_train_torch PRIVATE flash_train torch
                                                   "${FTRAIN_TORCH_PYTHON_LIBRARY}")

  # scikit-build-core collects the wheel from these install rules.
  install(TARGETS _flash_train_torch LIBRARY DESTINATION flash_train/torch)
endif()
