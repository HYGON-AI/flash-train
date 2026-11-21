import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class MultirankLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="moe launcher ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.worker = self.root / "fake worker"
        self.worker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys, time\n"
            "mode = os.environ['MOE_LAUNCHER_TEST_MODE']\n"
            "gtest = sys.argv[1].startswith('--gtest_filter=')\n"
            "def report(result='COMPLETED'):\n"
            "    name = sys.argv[1].split('=', 1)[1]\n"
            "    suite, case = name.split('.', 1)\n"
            "    path = pathlib.Path(next(arg.split('json:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output=')))\n"
            "    if mode == 'missing_report': return\n"
            "    if mode == 'malformed_report':\n"
            "        path.write_text('not json')\n"
            "        return\n"
            "    document = {'tests': 1, 'failures': 0, 'testsuites': [{'name': suite, 'testsuite': "
            "[{'name': case, 'status': 'RUN', 'result': result}]}]}\n"
            "    if mode == 'empty_filter': document.update(tests=0, testsuites=[])\n"
            "    if mode == 'wrong_case': document['testsuites'][0]['testsuite'][0]['name'] = 'WrongCase'\n"
            "    if mode == 'result_failure': document['failures'] = 1\n"
            "    path.write_text(json.dumps(document))\n"
            "preflight = sys.argv[1] == '--check-devices' or (gtest and 'DevicesAvailable' in sys.argv[1])\n"
            "if preflight:\n"
            "    if mode == 'preflight_timeout': time.sleep(60)\n"
            "    if gtest and mode != 'preflight_error':\n"
            "        report('SKIPPED' if mode == 'skip' else 'COMPLETED')\n"
            "        sys.exit(0)\n"
            "    sys.exit(77 if mode == 'skip' else 2 if mode == 'preflight_error' else 0)\n"
            "rank = int(os.environ['FTRAIN_TEST_MOE_RANK']) if gtest else int(sys.argv[4])\n"
            "root = pathlib.Path(os.environ['MOE_LAUNCHER_TEST_ROOT'])\n"
            "(root / f'pid-{rank}').write_text(str(os.getpid()))\n"
            "print('worker', rank, flush=True)\n"
            "if mode == 'fail' and rank == 0:\n"
            "    sys.exit(3)\n"
            "if mode in ('fail', 'timeout', 'interrupt'):\n"
            "    time.sleep(60)\n"
            "if gtest:\n"
            "    if mode == 'rank_missing_report': sys.exit(0)\n"
            "    report('SKIPPED' if mode == 'rank_skip' or (mode == 'partial_skip' and rank == 0) else 'COMPLETED')\n"
        )
        self.worker.chmod(0o700)
        self.command = [
            sys.executable, str(Path(__file__).with_name("run_multirank.py")),
            str(self.worker), "ag", "ag_nn_ring_chunk", str(self.root / "logs with spaces"),
            "--world-size", "2", "--timeout", "3",
        ]

    def environment(self, mode):
        return {**os.environ, "MOE_LAUNCHER_TEST_MODE": mode, "MOE_LAUNCHER_TEST_ROOT": str(self.root)}

    def launch(self, mode):
        return subprocess.run(self.command, env=self.environment(mode), capture_output=True, text=True, timeout=12)

    def assert_workers_stopped(self):
        for path in self.root.glob("pid-*"):
            with self.assertRaises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
        self.assertEqual(list(self.root.rglob("rendezvous-*")), [])

    def test_success_keeps_per_rank_logs(self):
        result = self.launch("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS ag", result.stdout)
        self.assertEqual(len(list(self.root.rglob("rank-*.log"))), 2)
        self.assert_workers_stopped()

    def test_python_worker_needs_no_executable_bit(self):
        self.worker.chmod(0o600)
        self.command.extend(["--interpreter", sys.executable])
        result = self.launch("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_workers_stopped()

    def test_gtest_success_keeps_structured_rank_results(self):
        self.command.append("--gtest")
        result = self.launch("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.root.rglob("rank-*.json"))), 2)
        self.assertEqual(len(list(self.root.rglob("preflight.json"))), 1)
        self.assert_workers_stopped()

    def test_gtest_skip_launches_no_ranks(self):
        self.command.append("--gtest")
        result = self.launch("skip")
        self.assertEqual(result.returncode, 77, result.stderr)
        self.assertEqual(list(self.root.glob("pid-*")), [])

    def test_gtest_rejects_invalid_preflight_reports(self):
        self.command.append("--gtest")
        for mode in ("missing_report", "malformed_report", "empty_filter", "wrong_case", "result_failure"):
            with self.subTest(mode=mode):
                result = self.launch(mode)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("invalid GoogleTest report", result.stderr)
                self.assertEqual(list(self.root.glob("pid-*")), [])

    def test_gtest_requires_rank_reports(self):
        self.command.append("--gtest")
        result = self.launch("rank_missing_report")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assert_workers_stopped()

    def test_gtest_rank_skips_are_not_reported_as_passes(self):
        self.command.append("--gtest")
        for mode, expected in (("rank_skip", 77), ("partial_skip", 1)):
            with self.subTest(mode=mode):
                result = self.launch(mode)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertNotIn("PASS ag", result.stdout)
                self.assert_workers_stopped()

    def test_gtest_rank_failure_cleans_up(self):
        self.command.append("--gtest")
        result = self.launch("fail")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("rank 0 exited with 3", result.stderr)
        self.assert_workers_stopped()

    def test_skip_launches_no_ranks(self):
        result = self.launch("skip")
        self.assertEqual(result.returncode, 77, result.stderr)
        self.assertEqual(list(self.root.glob("pid-*")), [])

    def test_preflight_error_is_not_a_skip(self):
        result = self.launch("preflight_error")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(list(self.root.glob("pid-*")), [])

    def test_preflight_timeout_launches_no_ranks(self):
        result = self.launch("preflight_timeout")
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(list(self.root.glob("pid-*")), [])

    def test_missing_binary_fails(self):
        self.worker.unlink()
        result = self.launch("success")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("FAIL:", result.stderr)

    def test_rank_failure_is_fail_fast_and_cleans_up(self):
        result = self.launch("fail")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("rank 0 exited with 3", result.stderr)
        self.assert_workers_stopped()

    def test_timeout_cleans_up(self):
        result = self.launch("timeout")
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assert_workers_stopped()

    def test_signal_cleans_up(self):
        process = subprocess.Popen(self.command, env=self.environment("interrupt"), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while len(list(self.root.glob("pid-*"))) != 2:
                if time.monotonic() >= deadline:
                    self.fail("workers did not start")
                time.sleep(0.02)
            process.terminate()
            process.communicate(timeout=6)
            self.assertEqual(process.returncode, 130)
            self.assert_workers_stopped()
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
