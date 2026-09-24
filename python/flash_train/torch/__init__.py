# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash-train PyTorch bindings."""

import ctypes
import os

# The extension module NEEDs libtorch.so, which only enters the process
# when torch itself is imported first, so the submodule imports torch
# itself and works on a bare `import flash_train.torch`.
import torch

# Wheels built with FTRAIN_BUILD_SHARED_LIBS=ON ship libflash_train.so
# beside the extension module. Loading it by absolute path pins the
# bundled copy before anything on LD_LIBRARY_PATH can shadow it, and
# registers its SONAME so the loader resolves the module's dependency
# against this copy. Static wheels merge the library into the extension
# and ship no file: the check below is a no-op, so one __init__.py serves
# both layouts.
_native_library = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "libflash_train.so")
if os.path.exists(_native_library):
    ctypes.CDLL(_native_library)

from ._flash_train_torch import gemm

__all__ = ["gemm"]
