# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Public API and packaging checks for the Hygon distributed backend."""

import os
from pathlib import Path

import flash_train.torch as flash_train_torch


def test_hygon_code_object_assets_are_installed():
    package_dir = Path(flash_train_torch.__file__).resolve().parent
    asset_dir = package_dir / "hygon_assets" / "gfx936"
    assert asset_dir.is_dir()
    assert any(asset_dir.glob("*.co"))
    assert not any(
        ("f" "lux") in path.name.lower() for path in asset_dir.glob("*.co")
    )
    assert not (package_dir / ("f" "lux_assets")).exists()


def test_pre_migration_aliases_are_removed():
    assert not hasattr(flash_train_torch, "init_" "f" "lux_shm")
    assert not hasattr(flash_train_torch, "Fl" "uxPlan")


def test_hygon_tuning_names_do_not_require_python_bridge(monkeypatch):
    monkeypatch.setenv("FTRAIN_HYGON_TEST_TUNING", "17")

    assert ("F" "LUX_TEST_TUNING") not in os.environ
    assert not hasattr(
        flash_train_torch, "_propagate_hygon_tuning_environment"
    )
