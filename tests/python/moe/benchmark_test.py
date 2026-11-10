import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))

from ftrain_bench.identity import row_key, workload_key, workload_label
from ftrain_bench.ops.moe.runner import _launch, aggregate, merge_ranks, pair_baselines
from ftrain_bench.ops import registry
from ftrain_bench.ops.moe import describe, partitions, routing_data, workload_label as moe_workload_label


def options(**changes):
    result = dict(route=None, world_size=8, experts=8, topk=2, partition="equal", balanced_topk="false",
                  seed=2026, n_chunks=0)
    result.update(changes)
    return result


def test_moe_registry_and_global_shapes():
    assert "gemm" in registry()
    assert len(registry()["ag_group_gemm"]["routes"]) == 8
    assert len(registry()["moe_reduce_rs"]["routes"]) == 2
    assert callable(registry()["ag_group_gemm"]["benchmark_runner"])
    assert callable(registry()["moe_reduce_rs"]["benchmark_runner"])
    ag = describe("ag_group_gemm", (128, 1024, 2048), options())
    rs = describe("moe_reduce_rs", (128, 8192, 4096), options())
    assert (ag["operator_m"], ag["operator_k"], ag["operator_n"], ag["operator_topk"]) == (256, 1024, 256, 1)
    assert (rs["operator_m"], rs["operator_k"], rs["operator_n"], rs["operator_topk"]) == (128, 1024, 4096, 2)


@pytest.mark.parametrize("native", [False, True])
def test_common_drivers_dispatch_registered_runner(monkeypatch, native):
    from ftrain_bench import c_harness, runner

    calls = []

    def benchmark_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return "result.json"

    driver = c_harness if native else runner
    monkeypatch.setattr(driver, "get_op", lambda name: {"benchmark_runner": benchmark_runner})
    configuration = {"world_size": 8}
    if native:
        result = driver.run_c("custom", "smoke", None, "bf16", "bench", ["baseline.json"], "out",
                              options=configuration)
    else:
        result = driver.run("custom", "smoke", None, "bf16", "out", options=configuration)
    assert result == "result.json"
    assert calls[0][0] == ("custom", "smoke", None, "bf16", "out")
    assert calls[0][1]["options"] is configuration
    if native:
        assert calls[0][1]["bin_path"] == "bench"
        assert calls[0][1]["baseline"] == ["baseline.json"]


def test_common_workload_labels_are_operator_independent():
    assert workload_label({}) == ""
    assert workload_label({"workload": {"batch": 4}}) == "batch=4"
    assert workload_label({"workload": {"batch": 4}, "workload_label": "four batches"}) == "four batches"


def test_report_modules_do_not_import_torch_or_operator_packages():
    benchmarks = Path(__file__).resolve().parents[3] / "benchmarks"
    subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(benchmarks)!r}); "
                    "from ftrain_bench import report, curve; "
                    "assert 'torch' not in sys.modules; assert 'ftrain_bench.ops' not in sys.modules"],
                   check=True, capture_output=True, text=True)


def test_partitions_and_route_constraints():
    sizes = partitions(128, 8, "ragged")
    assert sum(sizes) == 128 and min(sizes) > 0 and len(set(sizes)) == 8
    with pytest.raises(ValueError, match="equal"):
        describe("ag_group_gemm", (128, 1024, 2048), options(route="ag_nt_a2a_chunk", partition="ragged"))
    with pytest.raises(ValueError, match="K=32768"):
        describe("ag_group_gemm", (32768, 1024, 2048), options(route="ag_nt_a2a_chunk_32k"))
    with pytest.raises(ValueError, match="does not belong"):
        describe("moe_reduce_rs", (128, 8192, 4096), options(route="ag_nn_ring_chunk"))


def test_routing_is_deterministic_rank_expert_major():
    workload = describe("ag_group_gemm", (128, 1024, 2048), options(partition="ragged"))
    ids, rows = routing_data((128, 1024, 2048), workload)
    assert (ids, rows) == routing_data((128, 1024, 2048), workload)
    assert len(ids) == len(rows) == 256
    offset = 0
    for size in workload["rank_m_sizes"]:
        assert ids[offset:offset + size * 2] == sorted(ids[offset:offset + size * 2])
        assert sorted(rows[offset:offset + size * 2]) == [row for row in range(offset // 2, offset // 2 + size) for _ in range(2)]
        offset += size * 2


def test_critical_statistics_do_not_pool_rank_timings():
    stats = aggregate([{"samples_ms": [1, 9]}, {"samples_ms": [7, 2]}])
    assert stats["samples_ms"] == [7, 9]
    assert stats["mean_ms"] == 8
    assert stats["pooled"]["mean_ms"] == 4.75
    with pytest.raises(ValueError, match="same iteration"):
        aggregate([{"samples_ms": [1]}, {"samples_ms": [1, 2]}])
    with pytest.raises(ValueError, match="finite"):
        aggregate([{"samples_ms": [float("nan")]}])


@pytest.mark.parametrize("field,value", [("route", "ag_nn_ring"), ("world_size", 4), ("seed", 1),
                                        ("rank_m_sizes", [8, 24] * 4), ("balanced_topk", True)])
def test_result_identity_contains_workload(field, value):
    row = {"op": "ag_group_gemm", "precision": "bf16", "shape": [128, 1024, 2048],
           "workload": describe("ag_group_gemm", (128, 1024, 2048), options())}
    other = copy.deepcopy(row)
    other["workload"][field] = value
    assert row_key(row) != row_key(other)
    assert workload_key(row) != workload_key(other)


def test_pairing_requires_identical_inputs(tmp_path):
    row = {"op": "ag_group_gemm", "precision": "bf16", "shape": [128, 1024, 2048],
           "workload": describe("ag_group_gemm", (128, 1024, 2048), options()),
           "tier": "convenience-c", "input_fingerprints": [["one"]], "ftrain": {"median_ms": 1}}
    reference = {**row, "tier": "convenience-python", "baseline": {"median_ms": 2}}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"meta": {"timestamp": "2026-11-09T00:00:00+00:00"}, "results": [reference]}))
    pair_baselines([row], [str(path)])
    assert row["speedup"] == 2
    candidate = {key: value for key, value in row.items() if key not in ("baseline", "speedup")}
    pair_baselines([candidate], [str(path)], operator_environment={"FTRAIN_MOE_TEST_TUNING": "1"})
    assert "baseline" not in candidate
    row["input_fingerprints"] = [["different"]]
    with pytest.raises(ValueError, match="fingerprints"):
        pair_baselines([row], [str(path)])


def test_launcher_failure_cleanup_and_logs(tmp_path):
    binary = tmp_path / "fail.py"
    binary.write_text("#!/usr/bin/env python3\nimport sys\nprint('owned failure', file=sys.stderr)\nsys.exit(3)\n")
    binary.chmod(0o755)
    config = {"op": "ag_group_gemm", "shape": [128, 1024, 2048], "warmup": 0, "iters": 1,
              "workload": describe("ag_group_gemm", (128, 1024, 2048), options(world_size=2))}
    with pytest.raises(RuntimeError, match="rank"):
        _launch(config, tmp_path, binary, 5)
    assert "owned failure" in (tmp_path / "rank-0.log").read_text()
    assert not list(tmp_path.glob("rendezvous-*"))


def test_launcher_keeps_gloo_stdout_out_of_result_json(tmp_path):
    binary = tmp_path / "worker.py"
    binary.write_text("#!/usr/bin/env python3\nprint('[Gloo] started')\nprint('{\"rows\":[')\n"
                      "print('[moe] dispatch diagnostics')\nprint(']}')\nprint('[Gloo] stopped')\n")
    binary.chmod(0o755)
    config = {"op": "ag_group_gemm", "shape": [128, 1024, 2048], "warmup": 0, "iters": 1,
              "workload": describe("ag_group_gemm", (128, 1024, 2048), options(world_size=2))}
    assert _launch(config, tmp_path, binary, 5) == [{"rows": []}, {"rows": []}]
    assert json.loads((tmp_path / "rank-0.json").read_text()) == {"rows": []}


def test_launcher_timeout_reaps_only_owned_workers(tmp_path):
    binary = tmp_path / "sleep.py"
    binary.write_text("#!/usr/bin/env python3\nimport os, pathlib, time\n"
                      "pathlib.Path(os.environ['FTRAIN_BENCH_RENDEZVOUS']).parent.joinpath("
                      "'pid-' + os.environ['FTRAIN_BENCH_RANK']).write_text(str(os.getpid()))\ntime.sleep(30)\n")
    binary.chmod(0o755)
    config = {"op": "ag_group_gemm", "shape": [128, 1024, 2048], "warmup": 0, "iters": 1,
              "workload": describe("ag_group_gemm", (128, 1024, 2048), options(world_size=2))}
    with pytest.raises(TimeoutError):
        _launch(config, tmp_path, binary, 0.5)
    pids = [int(path.read_text()) for path in tmp_path.glob("pid-*")]
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not list(tmp_path.glob("rendezvous-*"))


def test_rank_merger_rejects_incomplete_samples():
    config = {"op": "ag_group_gemm", "shape": [128, 1024, 2048], "native": False, "iters": 2,
              "workload": describe("ag_group_gemm", (128, 1024, 2048), options())}
    row = {"tier": "convenience-python", "impl": "AG", "shape": config["shape"], "fingerprint": ["sample"] * 4,
           "stats": {"samples_ms": [1, 2]}, "baseline": {"samples_ms": [4, 5]}}
    merged = merge_ranks(config, [{"rows": [copy.deepcopy(row)]} for _ in range(8)])
    assert merged[0]["ftrain"]["samples_ms"] == [1, 2]
    assert len(merged[0]["input_fingerprints"]) == 8
    assert workload_label(merged[0]) == moe_workload_label(config["workload"])
    with pytest.raises(ValueError, match="incomplete rank"):
        merge_ranks(config, [{"rows": [row]}])
    row["stats"]["samples_ms"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        merge_ranks(config, [{"rows": [row]}] * 8)


@pytest.mark.parametrize("legacy", [False, True])
def test_report_and_curve_preserve_distinct_moe_workloads(tmp_path, monkeypatch, legacy):
    from ftrain_bench import curve, report

    monkeypatch.setattr(report, "_chart", lambda *args: None)
    monkeypatch.setattr(curve, "_line_chart", lambda *args: None)
    rows = []
    for route in ("ag_nn_ring", "ag_nn_ring_chunk"):
        rows.append({"op": "ag_group_gemm", "precision": "bf16", "shape": [128, 1024, 2048],
                     "impl": "AG", "tier": "convenience-python", "ftrain": {"median_ms": 1},
                     "baseline": {"median_ms": 2}, "speedup": 2,
                     "workload": describe("ag_group_gemm", (128, 1024, 2048), options(route=route))})
        if not legacy:
            rows[-1]["workload_label"] = moe_workload_label(rows[-1]["workload"])
    meta = {"timestamp": "2026-11-09T00:00:00+00:00", "device": {}, "torch_version": "test",
            "wheel_version": "test", "git_sha": "test"}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"meta": meta, "results": rows}))
    text = Path(report.render([str(path)], str(tmp_path))).read_text()
    assert "ag_nn_ring /" in text and "ag_nn_ring_chunk /" in text
    curve_path = curve.curve([str(path)], str(tmp_path))
    text = Path(curve_path).read_text()
    assert "ag_nn_ring /" in text and "ag_nn_ring_chunk /" in text
