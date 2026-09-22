# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""基准执行器：跑 形状×基线×被测 矩阵并落盘 JSON（schema: ftrain-bench/1）。"""

import json
import os

import torch

from . import env as env_mod
from . import timing
from .ops import get_op
from .suites import get_suite

SCHEMA = "ftrain-bench/1"

TIMING_POLICY = {
    "warmup": 10,
    "iters": "adaptive-200ms",
    "bounds_iters": [10, 200],
    "stat": "median",
    "l2_flush": False,
}


def _select_device(torch):
    """FTRAIN_BENCH_DEVICE 指定设备号；跑基准前应确认所选卡空闲（计时对并发负载敏感）。"""
    dev = os.environ.get("FTRAIN_BENCH_DEVICE")
    if dev is not None and dev != "":
        torch.cuda.set_device(int(dev))


def _parse_shapes(text):
    shapes = []
    for part in text.split(","):
        dims = [int(x) for x in part.strip().lower().split("x")]
        if len(dims) != 3 or any(d <= 0 for d in dims):
            raise SystemExit(f"bad shape: {part!r} (expected m x k x n)")
        shapes.append(tuple(dims))
    return shapes


def _shape_label(shape):
    return "x".join(str(d) for d in shape)


def _print_row(row):
    label = _shape_label(row["shape"])
    if "error" in row:
        print(f"{label:>22}  ERROR  {row['error']}")
        return
    base = row["baseline"]["median_ms"]
    ftr = row["ftrain"]["median_ms"]
    print(f"{label:>22}  base {base:>10.3f} ms   ftrain {ftr:>10.3f} ms   x{row['speedup']:.2f}")


def _stamp(timestamp):
    return timestamp.split("+")[0].replace("-", "").replace(":", "")


def run(op_name, suite, shapes_text, precision, out_dir):
    op = get_op(op_name)
    if precision is None:
        precision = op["precisions"][0]
    if precision not in op["dtypes"]:
        raise SystemExit(
            f"unsupported precision: {precision} (available: {', '.join(op['precisions'])})"
        )
    shapes = _parse_shapes(shapes_text) if shapes_text else list(get_suite(op_name, suite))
    dtype = op["dtypes"][precision]

    _select_device(torch)
    meta = env_mod.collect(torch, mode="convenience-python")
    meta["timing"] = TIMING_POLICY

    print(f"# {op_name} [{precision}] mode=convenience-python")
    print(f"# device={meta['device'].get('name')} arch={meta['device'].get('arch')} "
          f"torch={meta['torch_version']} wheel={meta['wheel_version']}")
    print(f"# shapes={len(shapes)} from {'custom' if shapes_text else suite}")

    results = []
    for shape in shapes:
        tensors = op["make_args"](shape, dtype)
        row = {
            "op": op_name,
            "precision": precision,
            "shape": list(shape),
            "impl": op["ftrain"]["name"],
        }
        try:
            base = timing.measure(torch, lambda: op["baseline"]["call"](tensors))
            ftr = timing.measure(torch, lambda: op["ftrain"]["call"](tensors))
            row["baseline"] = {"name": op["baseline"]["name"], **base}
            row["ftrain"] = ftr
            row["speedup"] = round(base["median_ms"] / ftr["median_ms"], 3)
        except Exception as exc:  # 单形状失败不中断整轮，落盘记录错误
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        _print_row(row)
        del tensors
        torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{op_name}-{precision}-{_stamp(meta['timestamp'])}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "meta": meta, "results": results}, fh, indent=2)
    print(f"# written: {path}")
    return path
