# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""报告生成：结果 JSON → markdown 对比表 +（可选 matplotlib）柱状图。

图表依赖 matplotlib，未安装时只输出表格，不报错。
"""

import glob
import json
import os


def _load(paths):
    docs = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        if not matched:
            raise SystemExit(f"no result files match: {pattern}")
        for path in matched:
            with open(path, encoding="utf-8") as fh:
                docs.append(json.load(fh))
    return docs


def _conditions(meta):
    dev = meta["device"]
    return (
        f"{dev.get('name', '?')}（{dev.get('arch', '?')}） · "
        f"torch {meta['torch_version']} · ftrain-torch {meta['wheel_version']} · "
        f"{meta['timestamp'][:10]} · 便利层端到端"
    )


def _chart(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # 中文字体存在则用中文标签，否则回退英文，避免图中出现缺字方框
    available = {f.name for f in fm.fontManager.ttflist}
    cjk = next(
        (
            name
            for name in (
                "Noto Sans CJK SC",
                "WenQuanYi Zen Hei",
                "WenQuanYi Micro Hei",
                "Source Han Sans SC",
                "Microsoft YaHei",
            )
            if name in available
        ),
        None,
    )
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        labels = {"base": "PyTorch 组合实现", "y": "中位数耗时 (ms, log)"}
    else:
        labels = {"base": "PyTorch composite", "y": "median latency (ms, log)"}

    rows = sorted(rows, key=lambda r: r["shape"][0] * r["shape"][1] * r["shape"][2])
    shape_labels = ["x".join(str(d) for d in r["shape"]) for r in rows]
    base = [r["baseline"]["median_ms"] for r in rows]
    ftr = [r["ftrain"]["median_ms"] for r in rows]
    xs = range(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([x - width / 2 for x in xs], base, width, label=labels["base"], color="#9AA1AB")
    ax.bar([x + width / 2 for x in xs], ftr, width, label="flash-train", color="#C8402F")
    ax.set_yscale("log")
    ax.set_ylabel(labels["y"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(shape_labels, rotation=30, ha="right", fontsize=8)
    for x, b, f in zip(xs, base, ftr):
        ax.text(x - width / 2, b * 1.15, f"{b:.3g}", ha="center", fontsize=7, color="#5B626B")
        ax.text(x + width / 2, f * 1.15, f"{f:.3g}", ha="center", fontsize=7, color="#C8402F")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render(inputs, out_dir):
    docs = _load(inputs)
    newest = max(docs, key=lambda d: d["meta"]["timestamp"])
    rows = [r for d in docs for r in d["results"] if "error" not in r]
    if not rows:
        raise SystemExit("no successful rows in inputs")

    op_name = rows[0]["op"]
    precision = rows[0]["precision"]
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, f"{op_name}-{precision}.png")
    chart = _chart(rows, chart_path)

    lines = [
        f"# {op_name} 基准（{precision}）",
        "",
        f"> 条件：{_conditions(newest['meta'])}",
        f"> 口径：事件计时中位数；基线 {rows[0]['baseline']['name']}（PyTorch 组合实现）；"
        "计时策略与公平性规则见 [benchmarks/README.md](../../benchmarks/README.md)",
        "",
        "| 形状 (m×k×n) | 基线 ms | flash-train ms | 加速比 |",
        "|---|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r["shape"][0] * r["shape"][1] * r["shape"][2]):
        lines.append(
            f"| {'x'.join(str(d) for d in r['shape'])} "
            f"| {r['baseline']['median_ms']:.3f} "
            f"| {r['ftrain']['median_ms']:.3f} "
            f"| x{r['speedup']:.2f} |"
        )
    if chart:
        lines += ["", f"![{op_name} 对比（中位数耗时，log 轴）]({os.path.basename(chart)})"]
    lines += [
        "",
        "数据来源：`benchmarks/results/`（随版本提交；复现步骤见"
        " [benchmarks/README.md](../../benchmarks/README.md)）。",
        "",
    ]
    md_path = os.path.join(out_dir, f"{op_name}-{precision}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"# report: {md_path}" + (f" + {chart}" if chart else " (no chart: matplotlib missing)"))
    return md_path
