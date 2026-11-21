# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Python bindings: one gated section per framework switch; a new framework
# adds an option in the root plus a section here. The torch module uses separate
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
                      flash_train/torch/csrc/flash_train_torch_capi.cpp
                      flash_train/torch/csrc/flash_train_torch_moe.cpp)
  target_compile_definitions(_flash_train_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
  target_link_libraries(_flash_train_torch PRIVATE flash_train torch
                                                   "${FTRAIN_TORCH_PYTHON_LIBRARY}")

  # scikit-build-core collects the wheel from these install rules.
  install(TARGETS _flash_train_torch LIBRARY DESTINATION flash_train/torch)

  if(FTRAIN_BUILD_TESTS)
    set(FTRAIN_PYTHON_TEST_PACKAGE "${PROJECT_BINARY_DIR}/python-test-package")
    foreach(module IN ITEMS flash_train/__init__.py flash_train/torch/__init__.py flash_train/torch/moe.py)
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${module}" "${FTRAIN_PYTHON_TEST_PACKAGE}/${module}" COPYONLY)
    endforeach()
    add_custom_command(TARGET _flash_train_torch POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:_flash_train_torch>
                               "${FTRAIN_PYTHON_TEST_PACKAGE}/flash_train/torch/")
    if(FTRAIN_BUILD_SHARED_LIBS)
      add_custom_command(TARGET _flash_train_torch POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:flash_train>
                                 "${FTRAIN_PYTHON_TEST_PACKAGE}/flash_train/libflash_train.so")
    endif()
    add_test(NAME ftrain_torch_api
             COMMAND "${Python_EXECUTABLE}" -m pytest -q "${PROJECT_SOURCE_DIR}/tests/python/torch_gemm_test.py"
                     "${PROJECT_SOURCE_DIR}/tests/python/torch_moe_test.py"
                     "${PROJECT_SOURCE_DIR}/tests/python/moe/benchmark_test.py" -m "not multirank")
    set_tests_properties(ftrain_torch_api PROPERTIES LABELS "python;api" RESOURCE_LOCK ftrain_hip_devices
                         TIMEOUT 120
                         ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${FTRAIN_PYTHON_TEST_PACKAGE}")
    if(FTRAIN_TEST_MOE_MULTIRANK)
      foreach(route IN ITEMS ag_nn_a2a ag_nn_ring ag_nn_ring_chunk ag_nt_a2a ag_nt_a2a_chunk
                             ag_nt_ring_chunk ag_nt_a2a_chunk_32k ag_nt_ring_chunk_32k rs_nn rs_nt)
        add_test(NAME "ftrain_torch_moe_${route}"
                 COMMAND "${Python_EXECUTABLE}" -m pytest -q
                         "${PROJECT_SOURCE_DIR}/tests/python/torch_moe_test.py::test_moe_multirank_numerics[${route}]")
        set_tests_properties("ftrain_torch_moe_${route}" PROPERTIES LABELS "moe;python;multirank;numerics"
                             RESOURCE_LOCK ftrain_hip_devices PROCESSORS 8 TIMEOUT 400
                             SKIP_REGULAR_EXPRESSION "1 skipped"
                             ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${FTRAIN_PYTHON_TEST_PACKAGE}"
                             ENVIRONMENT "FTRAIN_TEST_MOE_MULTIRANK=1;FTRAIN_MOE_TEST_LOG_DIR=${PROJECT_BINARY_DIR}/tests/python-moe-numerics;OMP_NUM_THREADS=1;MKL_NUM_THREADS=1")
      endforeach()
    endif()
  endif()
endif()
