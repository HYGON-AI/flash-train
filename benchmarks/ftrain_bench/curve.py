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
    _chart_title,
    _conditions,
    _flops_key,
    _image_link,
    _load,
    _shape_label,
    _tier,
)


def _version_label(meta):
    # 同一 wheel 版本号会覆盖多个构建提交，拼上短 sha 才能区分曲线上的版本点
    wheel = meta.get("wheel_version", "")
    sha = meta.get("git_sha", "")
    parts = []
    if wheel and wheel != "unknown":
        parts.append(wheel)
    if sha and sha != "unknown":
        parts.append(sha[:8])
    return "+".join(parts) if parts else "?"


def _line_chart(series, versions, path, tier):
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
    ax.set_title(_chart_title(tier, labels["cjk"]), fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def curve(inputs, out_dir, pages_url=""):
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
        tier = _tier(row)
        # L4 同一形状下标枚举多个实现，按 (形状, 下标, 实现) 各成一条曲线
        ident = (tuple(row["shape"]), row.get("index"), row.get("impl"))
        display = _shape_label(row["shape"])
        if tier == "primitive":
            display = f"{display} #{row.get('index', 0)} {row.get('impl', '')}"
        series = by_tier.setdefault(tier, {}).setdefault(ident, {"display": display, "values": {}})
        series["values"][label] = row["ftrain"]["median_ms"]

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
        first_col = "形状 (m×k×n) · 实现" if tier == "primitive" else "形状 (m×k×n)"
        header = f"| {first_col} | " + " | ".join(versions) + " |"
        lines += [header, "|" + "---|" * (len(versions) + 1)]
        ordered = sorted(
            tier_map.items(),
            key=lambda item: (_flops_key({"shape": list(item[0][0])}), item[0][1] or 0),
        )
        for _ident, series in ordered:
            cells = [f"{series['values'].get(v, float('nan')):.3f}" for v in versions]
            lines.append(f"| {series['display']} | " + " | ".join(cells) + " |")
        chart = _line_chart(
            {s["display"]: s["values"] for s in tier_map.values()},
            versions,
            os.path.join(out_dir, f"{op_name}-{precision}-{tier}-curve.png"),
            tier,
        )
        if chart:
            lines += [
                "",
                f"![{TIER_TITLES[tier]} 曲线]({_image_link(pages_url, os.path.basename(chart))})",
            ]
        lines += [""]

    lines += [
        "数据来源：`benchmarks/results/`；口径与公平性规则见"
        " [benchmarks/README.md](https://github.com/HYGON-AI/flash-train/blob/develop/benchmarks/README.md)。",
        "",
    ]
    md_path = os.path.join(out_dir, f"{op_name}-{precision}-curve.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"# curve: {md_path} (versions: {', '.join(versions)})")
    return md_path
