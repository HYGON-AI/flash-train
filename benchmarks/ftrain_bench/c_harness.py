# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""C harness 编排：由标准套件生成命令行、收集 JSON 行、合并 Python 基线。

C 二进制只做测量（形状经参数传入、结果打到 stdout）；套件选择、环境元数据、
基线配对（同 shape 对齐 Python 层测得的 torch 组合实现）与落盘都在本模块完成。
"""

import json
import os
import subprocess

from . import env as env_mod
from .ops import get_op
from .runner import TIMING_POLICY, _parse_shapes, _stamp
from .suites import get_suite

TIERS_WITH_BASELINE = ("convenience-c", "plan-reuse")

_TIER_TITLES = {
    "convenience-c": "L2 C 便利层（端到端）",
    "plan-reuse": "L3 Plan 复用（稳态）",
    "primitive": "L4 逐 Primitive（实现矩阵）",
}


def _load_baseline(paths):
    """接受一个或多个 L1 结果文件；同形状以时间戳最新的文档为准。"""
    if not paths:
        return {}
    docs = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            docs.append(json.load(fh))
    docs.sort(key=lambda d: d["meta"]["timestamp"])
    merged = {}
    for doc in docs:
        for r in doc["results"]:
            if "error" not in r:
                merged[tuple(r["shape"])] = r
    return merged


def run_c(op_name, suite, shapes_text, precision, bin_path, baseline, out_dir):
    op = get_op(op_name)
    if precision is None:
        precision = op["precisions"][0]
    shapes = _parse_shapes(shapes_text) if shapes_text else list(get_suite(op_name, suite))

    if not os.path.exists(bin_path):
        raise SystemExit(f"c harness not found: {bin_path} (build with -DFTRAIN_BUILD_BENCHMARKS=ON)")
    cmd = [os.path.abspath(bin_path), precision] + [",".join(str(d) for d in s) for s in shapes]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"c harness failed ({proc.returncode}):\n{proc.stderr}")

    base_by_shape = _load_baseline(baseline)
    rows = []
    for row in json.loads(proc.stdout)["rows"]:
        row["op"] = op_name
        row["precision"] = precision
        row["ftrain"] = row.pop("stats")
        if row["tier"] in TIERS_WITH_BASELINE and tuple(row["shape"]) in base_by_shape:
            brow = base_by_shape[tuple(row["shape"])]
            row["baseline"] = brow["baseline"]
            row["speedup"] = round(brow["baseline"]["median_ms"] / row["ftrain"]["median_ms"], 3)
        rows.append(row)

    import torch  # 仅元数据采集需要，运行环境必装

    meta = env_mod.collect(torch, mode="c-harness")
    meta["timing"] = TIMING_POLICY

    print(f"# {op_name} [{precision}] c-harness device={meta['device'].get('name')} "
          f"torch={meta['torch_version']}")
    for row in rows:
        label = "x".join(str(d) for d in row["shape"])
        title = _TIER_TITLES.get(row["tier"], row["tier"])
        extra = f" ws={row['workspace_bytes']}B" if "workspace_bytes" in row else ""
        speed = f" x{row['speedup']:.2f}" if "speedup" in row else ""
        print(f"{label:>22}  [{title}] {row['impl']}{extra} "
              f"{row['ftrain']['median_ms']:>10.3f} ms{speed}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{op_name}-{precision}-c-{_stamp(meta['timestamp'])}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": "ftrain-bench/1", "meta": meta, "results": rows}, fh, indent=2)
    print(f"# written: {path}")
    return path
