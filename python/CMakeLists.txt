# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# The PyTorch binding: two host-only translation units behind one pybind11
# module, _ftrain_torch, imported through the ftrain_torch python package.
# ftrain_torch.cpp lives in the torch world and converts at::Tensor
# arguments to plain pointers; ftrain_torch_capi.cpp lives in the
# HIP-flavored library world and calls the staged C API on torch's current
# stream. The split exists because the HCU torch wheel's CUDA-compat
# headers and the HIP runtime headers cannot coexist in one translation
# unit. libflash_train merges statically into the module.
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# This HCU TorchConfig defines no torch_python imported target; the
# pybind11<->at::Tensor casters live in that library.
find_library(FTRAIN_TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")

# NO_EXTRAS: pybind11's default LTO breaks GNU ld's static-archive
# member extraction on this toolchain, silently dropping libflash_train.
pybind11_add_module(_ftrain_torch NO_EXTRAS ftrain_torch.cpp ftrain_torch_capi.cpp)
target_compile_definitions(_ftrain_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
target_link_libraries(_ftrain_torch PRIVATE flash_train torch "${FTRAIN_TORCH_PYTHON_LIBRARY}")

# scikit-build-core collects the wheel from these install rules: the
# extension lands inside the python package directory.
install(TARGETS _ftrain_torch LIBRARY DESTINATION ftrain_torch)
