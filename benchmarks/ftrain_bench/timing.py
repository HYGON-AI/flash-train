# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""事件计时与统计：预热后按目标时长自适应迭代数，主口径为中位数。

基线与被测走同一计时路径（torch 当前流上事件计时），公平性优先于绝对值。
"""

import statistics


def _time_once(torch, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def measure(torch, fn, warmup=10, target_ms=200.0, min_iters=10, max_iters=200):
    """测量 fn 的稳态耗时分布，返回统计字典（单位 ms）。

    fn 必须是可在当前流上重复提交的调用；缓冲区复用由调用方保证，
    因此测得的是 steady state 而非冷启动。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    est = max(_time_once(torch, fn), 1e-3)
    iters = max(min_iters, min(max_iters, int(target_ms / est) + 1))
    samples = sorted(_time_once(torch, fn) for _ in range(iters))

    p10 = samples[max(0, int(len(samples) * 0.10) - 1)]
    p90 = samples[min(len(samples) - 1, int(len(samples) * 0.90))]
    return {
        "median_ms": round(statistics.median(samples), 6),
        "mean_ms": round(statistics.fmean(samples), 6),
        "min_ms": round(samples[0], 6),
        "p10_ms": round(p10, 6),
        "p90_ms": round(p90, 6),
        "iters": iters,
    }
