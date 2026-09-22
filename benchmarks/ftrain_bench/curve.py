# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""跨版本性能曲线：聚合 results/ 历史 JSON，按 (算子, 精度, tier, 形状) 串联版本序列。

单一版本时输出快照表与占位说明；版本随 release 积累后曲线自动延长。
"""

import json
import os

from .report import (
    TIER_ORDER,
    TIER_TITLES,
    _chart_labels,
    _conditions,
    _flops_key,
    _load,
    _shape_label,
    _tier,
)


def _version_label(meta):
    wheel = meta.get("wheel_version", "unknown")
    if wheel and wheel != "unknown":
        return wheel
    sha = meta.get("git_sha", "")
    return sha[:8] if sha and sha != "unknown" else "?"


def _line_chart(series, versions, path, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = _chart_labels()
    xs = range(len(versions))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for shape_label, version_map in series.items():
        medians = [version_map.get(v, float("nan")) for v in versions]
        ax.plot(list(xs), medians, marker="o", label=shape_label)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel(labels["y"])
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def curve(inputs, out_dir):
    docs = _load(inputs)
    docs.sort(key=lambda d: d["meta"]["timestamp"])
    flat = [
        (_version_label(d["meta"]), r)
        for d in docs
        for r in d["results"]
        if "error" not in r
    ]
    if not flat:
        raise SystemExit("no successful rows in inputs")

    op_name = flat[0][1]["op"]
    precision = flat[0][1]["precision"]
    versions = list(dict.fromkeys(label for label, _ in flat))

    by_tier = {}
    for label, row in flat:
        tier_map = by_tier.setdefault(_tier(row), {})
        shape_map = tier_map.setdefault(_shape_label(row["shape"]), {})
        shape_map[label] = row["ftrain"]["median_ms"]

    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"# {op_name} 性能曲线（{precision}）",
        "",
        f"> 最近环境：{_conditions(docs[-1]['meta'])}；聚合全部历史结果"
        "（`benchmarks/results/`）自动生成。",
        "",
    ]
    if len(versions) == 1:
        lines += [
            f"> 当前仅一个版本（{versions[0]}）的数据——曲线将随版本发布自动延长。",
            "",
        ]

    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        tier_map = by_tier[tier]
        lines += ["## " + TIER_TITLES[tier], ""]
        header = "| 形状 (m×k×n) | " + " | ".join(versions) + " |"
        lines += [header, "|" + "---|" * (len(versions) + 1)]
        ordered = sorted(
            tier_map.items(),
            key=lambda item: _flops_key({"shape": [int(x) for x in item[0].split("x")]}),
        )
        for shape_label, version_map in ordered:
            cells = [f"{version_map.get(v, float('nan')):.3f}" for v in versions]
            lines.append(f"| {shape_label} | " + " | ".join(cells) + " |")
        chart = _line_chart(
            tier_map,
            versions,
            os.path.join(out_dir, f"{op_name}-{precision}-{tier}-curve.png"),
            TIER_TITLES[tier],
        )
        if chart:
            lines += ["", f"![{TIER_TITLES[tier]} 曲线]({os.path.basename(chart)})"]
        lines += [""]

    lines += [
        "数据来源：`benchmarks/results/`；口径与公平性规则见"
        " [benchmarks/README.md](../../benchmarks/README.md)。",
        "",
    ]
    md_path = os.path.join(out_dir, f"{op_name}-{precision}-curve.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"# curve: {md_path} (versions: {', '.join(versions)})")
    return md_path
