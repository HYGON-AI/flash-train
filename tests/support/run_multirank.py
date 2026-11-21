import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def stop_processes(processes):
    owned = list(processes)
    processes.clear()
    for process in owned:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2
    for process in owned:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in owned:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt


def gtest_status(path, expected_name):
    try:
        document = json.loads(path.read_text())
        cases = [(suite["name"], case) for suite in document["testsuites"] for case in suite["testsuite"]]
        if document["tests"] != 1 or len(cases) != 1 or document["failures"] or document.get("errors", 0):
            raise ValueError("expected exactly one passing or skipped GoogleTest case")
        suite, case = cases[0]
        if f"{suite}.{case['name']}" != expected_name or case.get("failures") or case["status"] != "RUN":
            raise ValueError("unexpected GoogleTest case or execution status")
        if case["result"] == "SKIPPED":
            return 77
        if case["result"] != "COMPLETED":
            raise ValueError("GoogleTest case did not complete")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: invalid GoogleTest report {path}: {error}", file=sys.stderr)
        return 1


def gtest_command(command, name, report):
    return [*command, f"--gtest_filter={name}", f"--gtest_output=json:{report}", "--gtest_repeat=1"]


def run(arguments):
    output = Path(arguments.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_directory = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    print(f"MoE logs: {run_directory}", flush=True)
    binary = str(Path(arguments.binary).resolve())
    command = [arguments.interpreter, binary] if arguments.interpreter else [binary]
    environment = os.environ.copy()
    preflight_name = "MoeNumericsEnvironmentTest.DevicesAvailable"
    case_name = f"Routes/MoeNumericsTest.MatchesHostReference/{arguments.route}"
    if arguments.gtest:
        environment["FTRAIN_TEST_MOE_WORLD_SIZE"] = str(arguments.world_size)
        for name in ("FTRAIN_TEST_MOE_RANK", "FTRAIN_TEST_MOE_RENDEZVOUS", "GTEST_TOTAL_SHARDS",
                     "GTEST_SHARD_INDEX", "GTEST_SHARD_STATUS_FILE"):
            environment.pop(name, None)
    processes = []
    deadline = time.monotonic() + arguments.timeout
    try:
        with contextlib.ExitStack() as stack:
            preflight_log = stack.enter_context((run_directory / "preflight.log").open("w"))
            preflight_report = run_directory / "preflight.json"
            preflight_command = (gtest_command(command, preflight_name, preflight_report) if arguments.gtest else
                                 [*command, "--check-devices", str(arguments.world_size)])
            preflight = subprocess.Popen(
                preflight_command, env=environment,
                stdout=preflight_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(preflight)
            try:
                status = preflight.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                print("FAIL: device preflight timed out", file=sys.stderr)
                return 124
            preflight_log.flush()
            processes.remove(preflight)
            if status == 0 and arguments.gtest:
                status = gtest_status(preflight_report, preflight_name)
            if status != 0:
                print((run_directory / "preflight.log").read_text(), end="", flush=True)
                return 77 if status == 77 else 1
            rendezvous = stack.enter_context(tempfile.TemporaryDirectory(prefix="rendezvous-", dir=run_directory))
            stack.callback(stop_processes, processes)
            ranks = []
            for rank in range(arguments.world_size):
                log = stack.enter_context((run_directory / f"rank-{rank}.log").open("w"))
                rank_environment = environment.copy()
                if arguments.gtest:
                    rank_environment.update({"FTRAIN_TEST_MOE_RANK": str(rank),
                                             "FTRAIN_TEST_MOE_RENDEZVOUS": rendezvous})
                    rank_command = gtest_command(command, case_name, run_directory / f"rank-{rank}.json")
                else:
                    rank_command = [*command, arguments.kind, arguments.route, rendezvous,
                                    str(rank), str(arguments.world_size)]
                process = subprocess.Popen(
                    rank_command, env=rank_environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                processes.append(process)
                ranks.append(process)
            pending = set(range(arguments.world_size))
            skipped = []
            while pending:
                for rank in list(pending):
                    status = ranks[rank].poll()
                    if status is None:
                        continue
                    pending.remove(rank)
                    if status == 0 and arguments.gtest:
                        status = gtest_status(run_directory / f"rank-{rank}.json", case_name)
                        if status == 77:
                            skipped.append(rank)
                            continue
                    if status != 0:
                        print(f"FAIL: rank {rank} exited with {status}", file=sys.stderr)
                        log_text = (run_directory / f"rank-{rank}.log").read_text(errors="replace")
                        print("\n".join(log_text.splitlines()[-40:]), file=sys.stderr)
                        return 1
                if pending and time.monotonic() >= deadline:
                    print(f"FAIL: timed out waiting for ranks {sorted(pending)}", file=sys.stderr)
                    return 124
                if pending:
                    time.sleep(0.05)
            if skipped:
                if len(skipped) == arguments.world_size:
                    print("SKIP: all GoogleTest ranks skipped", flush=True)
                    return 77
                print(f"FAIL: inconsistent GoogleTest skips on ranks {skipped}", file=sys.stderr)
                return 1
            print(f"PASS {arguments.kind} {arguments.route} ranks={arguments.world_size}", flush=True)
            return 0
    except KeyboardInterrupt:
        print("FAIL: multirank test interrupted", file=sys.stderr)
        return 130
    except OSError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        stop_processes(processes)


def main():
    parser = argparse.ArgumentParser(description="Run isolated MoE correctness workers without Torch or MPI")
    parser.add_argument("binary")
    parser.add_argument("kind", choices=("ag", "rs"))
    parser.add_argument("route")
    parser.add_argument("output_directory")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--interpreter", help="Interpreter for a Python worker; omitted for native binaries")
    parser.add_argument("--gtest", action="store_true", help="Run parameterized MoE tests from ftrain_test")
    arguments = parser.parse_args()
    if not 2 <= arguments.world_size <= 8 or arguments.timeout <= 0:
        parser.error("world-size must be in [2, 8] and timeout must be positive")
    if arguments.gtest and not arguments.route.startswith(arguments.kind + "_"):
        parser.error("route must belong to the selected ag|rs operator")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run(arguments)


if __name__ == "__main__":
    sys.exit(main())
