# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash-train python package: operator bindings for training workloads."""

import importlib
import typing


def __getattr__(name: str) -> typing.Any:
    # The umbrella stays torch-free: `import flash_train` loads nothing
    # heavy, and `flash_train.torch` pulls in the binding (and torch with
    # it) on first attribute access. The submodule is imported directly; a
    # `from . import` statement here would probe this very attribute
    # through the import machinery's hasattr check and recurse forever.
    if name == "torch":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["torch"]
