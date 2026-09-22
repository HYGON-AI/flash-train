# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""标准形状集查询与导出；导出的 JSON 供后续 C harness 复用同一网格。"""

import json

from .ops import get_op, registry


def get_suite(op_name, suite_name):
    op = get_op(op_name)
    suites = op["suites"]
    if suite_name not in suites:
        raise SystemExit(
            f"unknown suite: {op_name}/{suite_name} (available: {', '.join(sorted(suites))})"
        )
    return suites[suite_name]


def export_json(path):
    payload = {name: op["suites"] for name, op in registry().items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
