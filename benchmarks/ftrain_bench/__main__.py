# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CLI 入口：python -m ftrain_bench run | report | export-suites"""

import argparse

from .report import render
from .runner import run
from .suites import export_json


def main():
    parser = argparse.ArgumentParser(
        prog="ftrain_bench", description="flash-train 基准套件（Python 层）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="运行基准并落盘 JSON")
    p_run.add_argument("--op", required=True, help="算子名，如 gemm")
    p_run.add_argument("--suite", default="standard", help="形状集名（默认 standard）")
    p_run.add_argument("--shapes", default=None, help="自定义形状，如 1024x4096x4096,256x256x256（m×k×n）")
    p_run.add_argument("--precision", default=None, help="精度，默认取算子首个支持项")
    p_run.add_argument("--out", default="results", help="结果输出目录")

    p_rep = sub.add_parser("report", help="由结果 JSON 生成 markdown 报告与图")
    p_rep.add_argument("--in", dest="inputs", nargs="+", required=True, help="结果 JSON 或通配符")
    p_rep.add_argument("--out", default="../docs/benchmarks", help="报告输出目录")

    p_exp = sub.add_parser("export-suites", help="导出标准形状集 JSON（供 C harness 复用）")
    p_exp.add_argument("--out", default="suites.json", help="输出文件")

    args = parser.parse_args()
    if args.cmd == "run":
        run(args.op, args.suite, args.shapes, args.precision, args.out)
    elif args.cmd == "report":
        render(args.inputs, args.out)
    else:
        export_json(args.out)


if __name__ == "__main__":
    main()
