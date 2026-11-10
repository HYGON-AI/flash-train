import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


ROUTES = (
    "ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a",
    "ag_nt_a2a_chunk", "ag_nt_ring_chunk", "rs_nn", "rs_nt",
    "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def telemetry(path):
    result = subprocess.run(["hy-smi", "--showuse", "--showmemuse", "--showclocks", "--showtemp"],
                            text=True, capture_output=True, check=False)
    path.write_text(result.stdout + result.stderr)


def gpu_processes():
    directory = Path("/sys/class/kfd/kfd/proc")
    processes = []
    for path in directory.iterdir():
        try:
            processes.append({"pid": int(path.name), "virtual_pid": int((path / "vpid").read_text())})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return processes


def wait_idle(args, directory):
    deadline = time.monotonic() + args.idle_timeout
    consecutive = 0
    with (directory / "idle-checks.jsonl").open("w") as output:
        while True:
            result = subprocess.run(["hy-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                                    text=True, capture_output=True, check=True)
            cards = json.loads(result.stdout)
            processes = gpu_processes()
            idle = len(cards) == 8 and not processes and all(
                float(card["HCU use (%)"]) == 0 and float(card["vram Total Used Memory (MiB)"]) < 512
                for card in cards.values())
            output.write(json.dumps({"timestamp_ns": time.time_ns(), "idle": idle,
                                     "cards": cards, "gpu_processes": processes}) + "\n")
            output.flush()
            consecutive = consecutive + 1 if idle else 0
            if consecutive >= 2:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Eight idle GPUs are required; other GPU contexts are still present")
            time.sleep(2)


def measure(args, route, position, variant):
    library = getattr(args, variant).resolve()
    shape = 32768 if route.endswith("32k") else args.shape
    directory = args.output / f"{route}-{position}-{variant}"
    directory.mkdir(parents=True)
    print(f"{route} {position} {variant}: waiting for eight idle GPUs", flush=True)
    wait_idle(args, directory)
    rendezvous = directory / "rendezvous"
    rendezvous.mkdir()
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("FTRAIN_"):
            del environment[name]
    environment.update({"HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                        "OMP_NUM_THREADS": "1",
                        "FTRAIN_MOE_RS_A2A_CU_PARTITION": "0",
                        "FTRAIN_MOE_RS_A2A_COMM_CU_COUNT": "0"})
    environment["LD_LIBRARY_PATH"] = str(library.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    command = [str(args.binary.resolve()), route, str(rendezvous.resolve()), "RANK", "8",
               str(shape), str(shape), str(shape), str(args.warmup), str(args.iters), args.partition,
               args.balanced_topk]
    metadata = {"variant": variant, "route": route, "position": position,
                "balanced_topk": route.startswith("ag_") and args.balanced_topk == "true",
                "unique_topk": False,
                "library": str(library), "library_sha256": digest(library),
                "binary_sha256": digest(args.binary), "command": command,
                "environment": {name: value for name, value in environment.items()
                                if name.startswith(("FTRAIN_", "HIP_", "CUDA_")) or
                                name in ("LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
                "started_ns": time.time_ns()}
    write_json(directory / "metadata.json", metadata)
    telemetry(directory / "telemetry-before.txt")
    processes = []
    logs = []
    observed_contexts = set()
    owned_contexts = set()
    unknown_since = {}
    resource_samples = []
    try:
        for rank in range(8):
            rank_command = [str(rank) if part == "RANK" else part for part in command]
            log = (directory / f"rank-{rank}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(rank_command, env=environment, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process in processes):
            failures = [process.returncode for process in processes if process.returncode not in (None, 0)]
            if failures:
                raise RuntimeError(f"Rank failed in {directory}: {failures}")
            owners = {process.pid for process in processes}
            contexts = gpu_processes()
            now = time.monotonic()
            resource_samples.append({"timestamp_ns": time.time_ns(), "gpu_processes": contexts})
            for context in contexts:
                observed_contexts.add(context["pid"])
                if context["pid"] in owners or context["virtual_pid"] in owners:
                    owned_contexts.add(context["pid"])
                else:
                    unknown_since.setdefault(context["pid"], now)
            external = [context for context in contexts if context["pid"] not in owned_contexts
                        and now - unknown_since[context["pid"]] >= 2]
            if external:
                metadata["external_gpu_processes"] = external
                raise RuntimeError(f"External GPU workload detected: {external}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Benchmark timed out: {directory}")
            time.sleep(0.1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(f"Rank failed in {directory}")
        if observed_contexts - owned_contexts:
            metadata["unidentified_gpu_contexts"] = sorted(observed_contexts - owned_contexts)
            raise RuntimeError(f"Unidentified GPU contexts were present: {metadata['unidentified_gpu_contexts']}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        metadata["ended_ns"] = time.time_ns()
        metadata["returncodes"] = [process.returncode for process in processes]
        metadata["rank_pids"] = [process.pid for process in processes]
        metadata["owned_gpu_contexts"] = sorted(owned_contexts)
        write_json(directory / "resource-samples.json", resource_samples)
        write_json(directory / "metadata.json", metadata)
        telemetry(directory / "telemetry-after.txt")
    ranks = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(8)]
    for rank in ranks:
        if not rank["passed"] or len(rank["gpu_ms"]) != args.iters or Path(rank["library"]) != library:
            raise RuntimeError(f"Validation or library provenance failed: {directory}")
        if rank["balanced_topk"] != metadata["balanced_topk"] or rank["unique_topk"]:
            raise RuntimeError(f"Routing flags differ from requested configuration: {directory}")
        mapped = {Path(name).resolve() for name in rank["mapped_libraries"]}
        implementations = {name for name in mapped if name.name.startswith("libflash_train.so")}
        if implementations != {library}:
            raise RuntimeError(f"Unexpected implementation linkage: {directory}")
        if args.baseline_backend is not None:
            backend = args.baseline_backend.resolve()
            if (backend in mapped) != (variant == "baseline"):
                raise RuntimeError(f"Unexpected external backend linkage: {directory}")
    metadata["mapped_library_sha256"] = {name: digest(name) for rank in ranks for name in rank["mapped_libraries"]}
    write_json(directory / "metadata.json", metadata)
    pooled = [value for rank in ranks for value in rank["gpu_ms"]]
    critical = [max(values) for values in zip(*(rank["gpu_ms"] for rank in ranks))]
    result = {"variant": variant, "position": position, "mean_ms": statistics.mean(pooled),
              "median_ms": statistics.median(pooled), "critical_mean_ms": statistics.mean(critical),
              "critical_median_ms": statistics.median(critical), "ranks": ranks}
    write_json(directory / "result.json", result)
    print(f"{route} {position} {variant}: mean={result['mean_ms']:.3f} "
          f"critical_mean={result['critical_mean_ms']:.3f} median={result['median_ms']:.3f}", flush=True)
    return result


def summarize(args, route, measurements):
    before = [run for run in measurements if run["variant"] == "baseline"]
    after = [run for run in measurements if run["variant"] == "candidate"]
    reference = measurements[0]["ranks"]
    for measurement in measurements:
        for rank, original in zip(measurement["ranks"], reference):
            if rank["fingerprints"] != original["fingerprints"]:
                raise RuntimeError(f"Inputs differ for {route}, rank {rank['rank']}")
            if any(abs(actual - expected) > 0.0005 + 0.025 * abs(expected)
                   for actual, expected in zip(rank["output_samples"], original["output_samples"])):
                raise RuntimeError(f"Outputs differ for {route}, rank {rank['rank']}")
    result = {"route": route, "partition": args.partition,
              "balanced_topk": reference[0]["balanced_topk"], "unique_topk": False,
              "measurements": measurements}
    for metric in ("mean_ms", "median_ms", "critical_mean_ms", "critical_median_ms"):
        baseline = statistics.mean(run[metric] for run in before)
        candidate = statistics.mean(run[metric] for run in after)
        result[metric] = {"baseline": baseline, "candidate": candidate,
                          "change_percent": (candidate / baseline - 1) * 100}
    result["within_threshold"] = all(result[metric]["change_percent"] <= args.threshold
                                     for metric in ("mean_ms", "critical_mean_ms"))
    write_json(args.output / f"summary-{route}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-backend", type=Path,
                        help="Optional external backend whose presence is required only for the baseline")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    parser.add_argument("--shape", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--idle-timeout", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--partition", choices=("equal", "ragged"), default="equal")
    parser.add_argument("--balanced-topk", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if args.cycles <= 0 or args.iters <= 0 or args.warmup < 0 or args.idle_timeout < 0:
        parser.error("cycles/iters must be positive; warmup/idle-timeout must be nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    results = []
    for route in args.routes:
        runs = [measure(args, route, position, variant) for position, variant in enumerate(
            ("baseline", "candidate", "candidate", "baseline") * args.cycles, 1)]
        results.append(summarize(args, route, runs))
        write_json(args.output / "summary.json", results)
    print("route baseline_mean candidate_mean change% critical_change% gate", flush=True)
    for result in results:
        metric = result["mean_ms"]
        critical = result["critical_mean_ms"]
        print(f"{result['route']} {metric['baseline']:.3f} {metric['candidate']:.3f} "
              f"{metric['change_percent']:+.2f} {critical['change_percent']:+.2f} "
              f"{'PASS' if result['within_threshold'] else 'RECHECK'}", flush=True)


if __name__ == "__main__":
    main()
