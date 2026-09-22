# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""运行环境采集：全部字段尽力而为，单字段失败不影响基准运行。"""

import datetime
import importlib.metadata
import os
import platform
import subprocess


def _git_sha():
    # 容器工作区通常不含 .git（同步时排除），由调用方经环境变量注入仓库 SHA
    override = os.environ.get("FTRAIN_BENCH_GIT_SHA")
    if override:
        return override
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return "unknown"


def _wheel_version():
    try:
        return importlib.metadata.version("ftrain-torch")
    except Exception:
        return "unknown"


def _device(torch):
    try:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(prop, "gcn_arch_name", "") or "unknown"
        return {
            "name": prop.name,
            "arch": arch,
            "count": torch.cuda.device_count(),
            "total_memory_gb": round(prop.total_memory / 2**30, 1),
        }
    except Exception:
        return {"name": "unknown", "arch": "unknown", "count": 0}


def _dtk_version():
    for path in ("/opt/dtk/version", "/opt/dtk/VERSION"):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                return text
        except OSError:
            continue
    return os.environ.get("DTK_HOME", "unknown")


def collect(torch, mode):
    """采集一次运行的全部环境元数据，与计时结果一起落盘。"""
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "wheel_version": _wheel_version(),
        "device": _device(torch),
        "dtk_version": _dtk_version(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None) or "unknown",
        "python_version": platform.python_version(),
        "mode": mode,
    }
