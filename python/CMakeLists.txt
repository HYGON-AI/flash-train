# The PyTorch binding: two host-only translation units behind one pybind11
# module, _flash_train_torch, imported as flash_train.torch. The split keeps
# the HCU torch wheel's CUDA-compat headers and the HIP runtime headers in
# separate translation units; they cannot coexist in one.
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
target_link_libraries(_flash_train_torch PRIVATE flash_train torch "${FTRAIN_TORCH_PYTHON_LIBRARY}")

# scikit-build-core collects the wheel from these install rules; a shared
# build also ships libflash_train.so beside the module.
install(TARGETS _flash_train_torch LIBRARY DESTINATION flash_train/torch)
if(FTRAIN_BUILD_SHARED_LIBS)
  install(TARGETS flash_train LIBRARY DESTINATION flash_train/torch)
endif()
