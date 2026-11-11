"""Single-node MoE orchestration, safe process ownership and rank-max statistics."""

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from ...identity import workload_key
from ...report import _doc_time, _load
from . import describe, workload_label


def summarize(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timings must be nonempty, finite and positive")
    ordered = sorted(samples)
    return {"median_ms": statistics.median(ordered), "mean_ms": statistics.fmean(ordered),
            "min_ms": ordered[0], "p10_ms": ordered[max(0, len(ordered) // 10 - 1)],
            "p90_ms": ordered[min(len(ordered) - 1, len(ordered) * 9 // 10)], "iters": len(ordered)}


def aggregate(rank_stats):
    counts = {len(stats["samples_ms"]) for stats in rank_stats}
    if len(counts) != 1:
        raise ValueError("ranks did not execute the same iteration count")
    for stats in rank_stats:
        summarize(stats["samples_ms"])
    critical = [max(values) for values in zip(*(stats["samples_ms"] for stats in rank_stats))]
    pooled = [value for stats in rank_stats for value in stats["samples_ms"]]
    return {**summarize(critical), "aggregation": "per-iteration-rank-max",
            "samples_ms": critical, "pooled": summarize(pooled), "ranks": rank_stats}


def _stop_owned(processes):
    owned = list(processes)
    processes.clear()
    for process in owned:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    for process in owned:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _launch(config, directory, bin_path, timeout):
    processes = []
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"benchmark interrupted by signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        with contextlib.ExitStack() as stack:
            rendezvous = stack.enter_context(tempfile.TemporaryDirectory(prefix="rendezvous-", dir=directory))
            stack.callback(_stop_owned, processes)
            for rank in range(config["workload"]["world_size"]):
                environment = os.environ.copy()
                environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                    "FTRAIN_BENCH_RANK": str(rank),
                                    "FTRAIN_BENCH_WORLD_SIZE": str(config["workload"]["world_size"]),
                                    "FTRAIN_BENCH_RENDEZVOUS": rendezvous,
                                    "FTRAIN_BENCH_ROUTE": config["workload"]["route"],
                                    "FTRAIN_BENCH_EXPERTS": str(config["workload"]["experts"]),
                                    "FTRAIN_BENCH_TOPK": str(config["workload"]["topk"]),
                                    "FTRAIN_BENCH_SEED": str(config["workload"]["seed"]),
                                    "FTRAIN_BENCH_RANK_M": ",".join(map(str, config["workload"]["rank_m_sizes"])),
                                    "FTRAIN_BENCH_BALANCED": str(int(config["workload"]["balanced_topk"])),
                                    "FTRAIN_BENCH_N_CHUNKS": str(config["workload"]["n_chunks"]),
                                    "FTRAIN_BENCH_WARMUP": str(config["warmup"]),
                                    "FTRAIN_BENCH_ITERS": str(config["iters"]),
                                    "FTRAIN_BENCH_TIMEOUT": str(int(timeout))})
                if bin_path:
                    command = [str(Path(bin_path).resolve()), config["op"], "bf16",
                               ",".join(map(str, config["shape"]))]
                else:
                    command = [sys.executable, "-m", "ftrain_bench.ops.moe.worker", str(config_path.resolve()), str(rank)]
                result = stack.enter_context((directory / f"rank-{rank}.stdout").open("w"))
                log = stack.enter_context((directory / f"rank-{rank}.log").open("w"))
                processes.append(subprocess.Popen(command, stdout=result, stderr=log, env=environment,
                                                  start_new_session=True))
            pending = set(range(len(processes)))
            deadline = time.monotonic() + timeout
            while pending:
                for rank in list(pending):
                    status = processes[rank].poll()
                    if status is None:
                        continue
                    pending.remove(rank)
                    if status:
                        raise RuntimeError(f"rank {rank} failed ({status}); see {directory / f'rank-{rank}.log'}")
                if pending and time.monotonic() > deadline:
                    raise TimeoutError(f"MoE benchmark timed out; logs: {directory}")
                if pending:
                    time.sleep(0.05)
            documents = []
            for rank in range(len(processes)):
                text = (directory / f"rank-{rank}.stdout").read_text()
                start = text.rfind('{"rows":')
                if start < 0:
                    raise ValueError(f"rank {rank} produced no result JSON; logs: {directory}")
                payload = "\n".join(line for line in text[start:].splitlines()
                                    if not line.lstrip().startswith(("[moe]", "[Gloo]")))
                document, _ = json.JSONDecoder().raw_decode(payload)
                (directory / f"rank-{rank}.json").write_text(json.dumps(document, indent=2) + "\n")
                documents.append(document)
            return documents
    finally:
        _stop_owned(processes)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def merge_ranks(config, documents):
    rows = []
    if len(documents) != config["workload"]["world_size"]:
        raise ValueError("incomplete rank results")
    expected_rows = 3 if config["native"] else 1
    if any(len(document["rows"]) != expected_rows for document in documents):
        raise ValueError("incomplete benchmark tiers")
    for index, first in enumerate(documents[0]["rows"]):
        expected_tier = ("convenience-c", "plan-reuse", "primitive")[index] if config["native"] else "convenience-python"
        if first["tier"] != expected_tier or first["shape"] != config["shape"]:
            raise ValueError("unexpected benchmark workload/tier")
        peers = [document["rows"][index] for document in documents]
        if any((peer["tier"], peer["impl"], peer["shape"]) !=
               (first["tier"], first["impl"], first["shape"]) for peer in peers):
            raise ValueError("rank workload/tier mismatch")
        if any(len(peer["stats"]["samples_ms"]) != config["iters"] for peer in peers):
            raise ValueError("incomplete timing samples")
        if any(len(peer["fingerprint"]) != 4 for peer in peers):
            raise ValueError("missing input fingerprints")
        row = {key: value for key, value in first.items() if key not in ("stats", "baseline", "fingerprint")}
        row.update({"op": config["op"], "precision": "bf16", "workload": config["workload"],
                    "workload_label": workload_label(config["workload"]),
                    "input_fingerprints": [peer["fingerprint"] for peer in peers],
                    "ftrain": aggregate([peer["stats"] for peer in peers])})
        if not config["native"]:
            if any(len(peer["baseline"]["samples_ms"]) != config["iters"] for peer in peers):
                raise ValueError("incomplete baseline timing samples")
            row["baseline"] = {"name": "torch.distributed + expert matmul + routing", **aggregate(
                [peer["baseline"] for peer in peers])}
            row["speedup"] = row["baseline"]["median_ms"] / row["ftrain"]["median_ms"]
        rows.append(row)
    return rows


def pair_baselines(rows, paths, *, operator_environment=None):
    if not paths:
        return
    candidates = {}
    for document in sorted(_load(paths), key=_doc_time):
        if operator_environment is not None and document["meta"].get("operator_environment", {}) != operator_environment:
            continue
        for row in document["results"]:
            if "error" not in row and row.get("tier") == "convenience-python" and "baseline" in row:
                candidates[workload_key(row)] = row
    for row in rows:
        baseline = candidates.get(workload_key(row))
        if baseline is None or row["tier"] == "primitive":
            continue
        if baseline.get("input_fingerprints") != row["input_fingerprints"]:
            raise ValueError("Python/C++ input fingerprints differ; refusing baseline pairing")
        row["baseline"] = baseline["baseline"]
        row["speedup"] = baseline["baseline"]["median_ms"] / row["ftrain"]["median_ms"]


def run_moe(op_name, suite, shapes_text, precision, out_dir, *, options=None, bin_path=None, baseline=None):
    import torch

    from ... import env as env_mod
    from ...runner import _parse_shapes, _stamp
    from ...suites import get_suite

    defaults = dict(route=None, world_size=8, experts=64, topk=6, partition="equal", balanced_topk="false",
                    seed=2026, n_chunks=0, warmup=10, iters=40, timeout=600)
    defaults.update(options or {})
    if precision not in (None, "bf16"):
        raise ValueError("MoE benchmarks support bf16 only")
    if defaults["iters"] <= 0 or defaults["warmup"] < 0 or defaults["timeout"] <= 0:
        raise ValueError("iters/timeout must be positive and warmup nonnegative")
    if not 0 <= defaults["seed"] < 2**32 or defaults["n_chunks"] < 0:
        raise ValueError("seed must fit uint32 and n_chunks must be nonnegative")
    shapes = _parse_shapes(shapes_text) if shapes_text else list(get_suite(op_name, suite))
    workloads = [describe(op_name, shape, defaults) for shape in shapes]
    if torch.cuda.device_count() < defaults["world_size"]:
        raise RuntimeError(f"requires {defaults['world_size']} visible devices")
    for rank in range(defaults["world_size"]):
        if not getattr(torch.cuda.get_device_properties(rank), "gcnArchName", "").startswith("gfx936"):
            raise RuntimeError("MoE benchmark requires gfx936 on every participating device")
    if bin_path and not Path(bin_path).is_file():
        raise FileNotFoundError(f"C harness not found: {bin_path}; build with FTRAIN_BUILD_BENCHMARKS=ON")
    meta = env_mod.collect(torch, mode="c-harness" if bin_path else "convenience-python")
    meta["device"]["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    meta["suite"] = "custom" if shapes_text else suite
    meta["timing"] = {"warmup": defaults["warmup"], "iters": defaults["iters"], "stat": "median",
                      "aggregation": "per-iteration-rank-max", "l2_flush": False,
                      "setup_included": False, "cpu_barriers_included": False}
    meta["visibility"] = {name: os.environ.get(name) for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    meta["operator_environment"] = {name: value for name, value in os.environ.items()
                                    if name.startswith("FTRAIN_MOE_")}
    meta["sampling"] = {"finite_scan": "all native output elements", "reference_points_per_rank": 128,
                        "absolute_tolerance": 0.0005, "relative_tolerance": 0.03}
    if bin_path:
        meta["binary"] = str(Path(bin_path).resolve())
        meta["binary_sha256"] = hashlib.sha256(Path(bin_path).read_bytes()).hexdigest()
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identifier = f"{op_name}-bf16{'-c' if bin_path else ''}-{_stamp(meta['timestamp'])}-{time.time_ns() % 1000000:06d}"
    logs = output / (identifier + "-logs")
    logs.mkdir()
    results = []
    for index, (shape, workload) in enumerate(zip(shapes, workloads)):
        directory = logs / str(index)
        directory.mkdir()
        config = {"op": op_name, "shape": list(shape), "workload": workload, "native": bool(bin_path),
                  "warmup": defaults["warmup"], "iters": defaults["iters"]}
        documents = _launch(config, directory, bin_path, defaults["timeout"])
        rows = merge_ranks(config, documents)
        pair_baselines(rows, baseline, operator_environment=meta["operator_environment"])
        results.extend(rows)
        for row in rows:
            print(f"{shape} {workload['route']} {row['tier']}: "
                  f"median={row['ftrain']['median_ms']:.3f} mean={row['ftrain']['mean_ms']:.3f} ms", flush=True)
    path = output / (identifier + ".json")
    path.write_text(json.dumps({"schema": "ftrain-bench/1", "meta": meta, "results": results}, indent=2) + "\n")
    print(f"# written: {path}")
    return str(path)
