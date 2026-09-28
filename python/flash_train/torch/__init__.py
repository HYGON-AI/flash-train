# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash-train PyTorch bindings."""

import ctypes
import os

# The extension NEEDs libtorch.so, which only enters the process with torch.
import torch

# Wheels built with FTRAIN_BUILD_SHARED_LIBS=ON ship libflash_train.so
# beside the extension module; loading it by absolute path pins the bundled
# copy over anything on LD_LIBRARY_PATH. Static wheels ship no file and
# skip this.
_native_library = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "libflash_train.so")
if os.path.exists(_native_library):
    ctypes.CDLL(_native_library)

from ._flash_train_torch import gemm

__all__ = ["gemm"]
