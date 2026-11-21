import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    groups = {}
    paths = sorted({path.resolve() for root in args.runs for path in root.glob("*/result.json")})
    if not paths:
        parser.error("No completed runs found")
    for path in paths:
        run = json.loads(path.read_text())
        rank = run["ranks"][0]
        key = (rank["route"], rank["ragged"], rank["logical_m"], rank["global_n"], rank["global_k"],
               rank["balanced_topk"], rank["unique_topk"])
        groups.setdefault(key, []).append((path, run))
    summaries = []
    for key, runs in sorted(groups.items()):
        reference = runs[0][1]["ranks"]
        for path, run in runs:
            if len(run["ranks"]) != 8:
                raise RuntimeError(f"Expected eight ranks: {path}")
            for rank, original in zip(run["ranks"], reference):
                if not rank["passed"] or rank["fingerprints"] != original["fingerprints"]:
                    raise RuntimeError(f"Validation/input mismatch: {path}")
                if any(abs(actual - expected) > 0.0005 + 0.025 * abs(expected)
                       for actual, expected in zip(rank["output_samples"], original["output_samples"])):
                    raise RuntimeError(f"A/B output mismatch: {path}")
        summary = dict(zip(("route", "ragged", "logical_m", "global_n", "global_k",
                            "balanced_topk", "unique_topk"), key))
        summary["source_results"] = [str(path) for path, run in runs]
        summary["sampled_outputs_equal"] = all(
            rank["output_samples"] == original["output_samples"] for path, run in runs
            for rank, original in zip(run["ranks"], reference))
        summary["maximum_reference_error"] = max(rank["max_abs"] for path, run in runs for rank in run["ranks"])
        for variant in ("baseline", "candidate"):
            selected = [(path, run) for path, run in runs if run["variant"] == variant]
            if not selected:
                raise RuntimeError(f"Missing {variant} measurements: {key}")
            hashes = {json.loads(path.with_name("metadata.json").read_text())["library_sha256"]
                      for path, run in selected}
            if len(hashes) != 1:
                raise RuntimeError(f"Multiple {variant} libraries in one comparison: {key}")
            pooled = [value for path, run in selected for rank in run["ranks"] for value in rank["gpu_ms"]]
            critical = [max(values) for path, run in selected
                        for values in zip(*(rank["gpu_ms"] for rank in run["ranks"]))]
            summary[variant] = {"launches": len(selected), "rank_samples": len(pooled),
                                "library_sha256": next(iter(hashes)), "mean_ms": statistics.mean(pooled),
                                "median_ms": statistics.median(pooled),
                                "critical_mean_ms": statistics.mean(critical),
                                "critical_median_ms": statistics.median(critical),
                                "equal_launch_weight_mean_ms": statistics.mean(run["mean_ms"] for path, run in selected)}
        summary["changes_percent"] = {
            metric: (summary["candidate"][metric] / summary["baseline"][metric] - 1) * 100
            for metric in ("mean_ms", "median_ms", "critical_mean_ms", "critical_median_ms",
                           "equal_launch_weight_mean_ms")}
        summary["within_threshold"] = all(summary["changes_percent"][metric] <= args.threshold
                                           for metric in ("mean_ms", "critical_mean_ms"))
        summaries.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"threshold_percent": args.threshold, "profiles": summaries}, indent=2) + "\n")
    for summary in summaries:
        print(f"{summary['route']:26s} {'ragged' if summary['ragged'] else 'equal':6s} "
              f"balanced_topk={str(summary['balanced_topk']).lower()} "
              f"{summary['baseline']['mean_ms']:.3f} -> {summary['candidate']['mean_ms']:.3f} ms "
              f"{summary['changes_percent']['mean_ms']:+.2f}% "
              f"critical={summary['changes_percent']['critical_mean_ms']:+.2f}% "
              f"{'PASS' if summary['within_threshold'] else 'RECHECK'}")


if __name__ == "__main__":
    main()
