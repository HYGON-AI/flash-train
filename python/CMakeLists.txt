# The PyTorch binding: two host-only translation units behind one pybind11
# module, _flash_train_torch, imported through the torch submodule of the
# flash_train python package (flash_train.torch). The sources live in
# flash_train/torch/csrc so the whole binding—python shell and C++ shell—
# stays in one place; wheel.exclude keeps them out of the wheel.
# flash_train_torch.cpp lives in the torch world and converts at::Tensor
# arguments to plain pointers; flash_train_torch_capi.cpp lives in the
# HIP-flavored library world and calls the staged C API on torch's current
# stream. The split exists because the HCU torch wheel's CUDA-compat
# headers and the HIP runtime headers cannot coexist in one translation
# unit. With FTRAIN_BUILD_SHARED_LIBS the module links the
# libflash_train.so shipped beside it in the package; otherwise
# libflash_train merges statically into the module.
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# This HCU TorchConfig defines no torch_python imported target; the
# pybind11<->at::Tensor casters live in that library.
find_library(FTRAIN_TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")

set(FTRAIN_TORCH_MODULE_EXTRAS "")
if(NOT FTRAIN_BUILD_SHARED_LIBS)
  # NO_EXTRAS: pybind11's default LTO breaks GNU ld's static-archive
  # member extraction on this toolchain, silently dropping libflash_train.
  # The shared build links a DSO instead of an archive, so the module may
  # keep the LTO extras.
  list(APPEND FTRAIN_TORCH_MODULE_EXTRAS NO_EXTRAS)
endif()

pybind11_add_module(_flash_train_torch ${FTRAIN_TORCH_MODULE_EXTRAS}
                    flash_train/torch/csrc/flash_train_torch.cpp
                    flash_train/torch/csrc/flash_train_torch_capi.cpp)
target_compile_definitions(_flash_train_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
target_link_libraries(_flash_train_torch PRIVATE flash_train torch "${FTRAIN_TORCH_PYTHON_LIBRARY}")

# scikit-build-core collects the wheel from these install rules: the
# extension lands inside the flash_train/torch package directory, and a
# shared build ships libflash_train.so beside it for __init__.py to
# preload.
install(TARGETS _flash_train_torch LIBRARY DESTINATION flash_train/torch)
if(FTRAIN_BUILD_SHARED_LIBS)
  install(TARGETS flash_train LIBRARY DESTINATION flash_train/torch)
endif()
