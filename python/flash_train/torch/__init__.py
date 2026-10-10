# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash-train PyTorch bindings."""

import ctypes
import os
from pathlib import Path

# The extension NEEDs libtorch.so, which only enters the process with torch.
import torch

# Wheels built with FTRAIN_BUILD_SHARED_LIBS=ON ship libflash_train.so in
# the package root; loading it by absolute path pins the bundled copy over
# anything on LD_LIBRARY_PATH. Static wheels ship no file and skip this.
_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_native_library = os.path.join(_package_root, "libflash_train.so")
if os.path.exists(_native_library):
    ctypes.CDLL(_native_library)

_package_dir = Path(__file__).resolve().parent
_code_object_dir = _package_dir / "hygon_assets" / "gfx936"
if _code_object_dir.is_dir():
    os.environ.setdefault("FTRAIN_HYGON_CODE_OBJECT_DIR", str(_code_object_dir))
    _effective_code_object_dir = os.environ["FTRAIN_HYGON_CODE_OBJECT_DIR"]
    os.environ.setdefault(
        "FTRAIN_HYGON_ASM_DIR_GFX936", _effective_code_object_dir + os.sep
    )
    os.environ.setdefault(
        "FTRAIN_HYGON_MOE_RS_ASM_DIR_GFX936",
        _effective_code_object_dir + os.sep,
    )

try:
    from ._flash_train_torch import gemm as _native_gemm
    from ._flash_train_torch import (
        close_ag_group_gemm_session,
        close_moe_reduce_rs_session,
        create_ag_group_gemm_plan as native_create_ag_group_gemm_plan,
        create_ag_group_gemm_session,
        create_moe_reduce_rs_plan as native_create_moe_reduce_rs_plan,
        create_moe_reduce_rs_session,
        execute_plan,
        get_plan_num_primitives,
        get_plan_primitive_name,
        init_hygon_shm,
    )
except ModuleNotFoundError as exc:
    # This shell ships even in wheels built without the binding; a missing
    # extension module is a build problem, not a missing dependency.
    if exc.name != "flash_train.torch._flash_train_torch":
        raise
    raise RuntimeError(
        "this wheel was built without the torch binding; rebuild it with "
        "FTRAIN_BUILD_TORCH_BINDINGS=ON"
    ) from exc


def gemm(*args, **kwargs):
    return _native_gemm(*args, **kwargs)


class HygonDistributedPlan:
    """Thin owner for a formal Plan capsule created by _flash_train_torch."""

    def __init__(self, capsule):
        self._capsule = capsule

    @property
    def num_primitives(self):
        return get_plan_num_primitives(self._capsule)

    def primitive_name(self, index=0):
        return get_plan_primitive_name(self._capsule, index) or None

    def execute(self, primitive_index=0):
        execute_plan(self._capsule, primitive_index)


def create_ag_group_gemm_plan(session_id, input, weight, topk_ids, output):
    return HygonDistributedPlan(
        native_create_ag_group_gemm_plan(session_id, input, weight, topk_ids, output)
    )


def create_moe_reduce_rs_plan(
    session_id, input, weight, chosen_experts, expert_weight, output
):
    return HygonDistributedPlan(
        native_create_moe_reduce_rs_plan(
            session_id, input, weight, chosen_experts, expert_weight, output
        )
    )


__all__ = [
    "gemm",
    "init_hygon_shm",
    "create_ag_group_gemm_session",
    "create_ag_group_gemm_plan",
    "close_ag_group_gemm_session",
    "create_moe_reduce_rs_session",
    "create_moe_reduce_rs_plan",
    "close_moe_reduce_rs_session",
    "HygonDistributedPlan",
]
