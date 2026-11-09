"""Offline checks; no Docker daemon, package install or GitHub writes required."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce = load("reproduce_ci", SCRIPTS / "reproduce-ci.py")
capture = load("capture_ci_env", SCRIPTS / "capture-ci-env.py")


def manifest():
    return {
        "schema": 1, "repository": "HYGON-AI/flash-train", "run_id": "123", "attempt": "2",
        "source": {"sha": "a" * 40, "tree": "b" * 40, "base_sha": "c" * 40,
                   "head_sha": "d" * 40, "base_url": "https://github.com/HYGON-AI/flash-train.git",
                   "head_url": "https://github.com/example/flash-train.git"},
        "image": {"digest": "registry.example/dtk@sha256:" + "e" * 64},
        "env": {"GITHUB_WORKSPACE": "/__w/flash-train/flash-train", "RUNNER_TEMP": "/__w/_temp",
                "TORCH_VERSION": "2.13.0", "PYTHON_VERSION": "3.12", "DTK_VER": "2610",
                "CPP_TESTS": "false", "CI_RUN_TESTS": "true", "PYPI_URL": "http://pypi.example:9929"},
        "aicc": {"url": "http://aicc.example/compiler.run", "sha256": "f" * 64, "version": "2.0.0+example"},
        "packages": {"torch": "2.13.0+dtk2610.example", "numpy": "1.26.4"},
        "setup_complete": True,
    }


class MetadataTests(unittest.TestCase):
    def test_requires_immutable_image_and_exact_aicc(self):
        reproduce.validate(manifest())
        for field in ("image", "aicc", "packages"):
            data = manifest()
            data.pop(field)
            with self.assertRaises(ValueError):
                reproduce.validate(data)

    def test_refuses_credentials_and_unsafe_mounts(self):
        data = manifest()
        data["env"]["PYPI_URL"] = "http://username:password@pypi.example"
        with self.assertRaises(ValueError):
            reproduce.validate(data)
        data = manifest()
        data["env"]["RUNNER_TEMP"] = "/"
        with self.assertRaises(ValueError):
            reproduce.validate(data)

    def test_aicc_selection_has_exact_url_version_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.txt"
            selection.write_text("完整版本: 2.0.0+dtk2610.example\n下载: http://aicc.example/compiler.run\n",
                                 encoding="utf-8")
            package = root / "compiler.run"
            package.write_bytes(b"exact installer")
            data = {}
            capture.capture_aicc(data, selection, package)
            self.assertEqual(data["aicc"]["sha256"], hashlib.sha256(b"exact installer").hexdigest())
            self.assertEqual(data["aicc"]["version"], "2.0.0+dtk2610.example")

    def test_records_actual_merge_parents_and_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(json.dumps({"pull_request": {"head": {
                "sha": "d" * 40, "repo": {"clone_url": "https://github.com/example/flash-train.git"}}}}))
            env = dict(manifest()["env"], GITHUB_REPOSITORY="HYGON-AI/flash-train", GITHUB_RUN_ID="123",
                       GITHUB_RUN_ATTEMPT="2", GITHUB_EVENT_NAME="pull_request", GITHUB_EVENT_PATH=str(event),
                       CI_CONTAINER_ID="container", GITHUB_TOKEN="do-not-save", PYPI_PASSWORD="do-not-save")
            def git(*args):
                return {("rev-parse", "HEAD"): "a" * 40, ("rev-parse", "HEAD^{tree}"): "b" * 40,
                        ("show", "-s", "--format=%P", "HEAD"): "c" * 40 + " " + "d" * 40}[args]
            with patch.dict(os.environ, env, clear=True), patch.object(capture, "git", side_effect=git), \
                    patch.object(capture, "docker_get", side_effect=[
                        {"Image": "image-id", "Config": {"Image": "registry.example/dtk:26.10"}},
                        {"RepoDigests": [manifest()["image"]["digest"]]}]):
                data = capture.capture_initial()
            self.assertEqual(data["source"]["base_sha"], "c" * 40)
            self.assertEqual(data["source"]["head_sha"], "d" * 40)
            self.assertNotIn("do-not-save", json.dumps(data))

    def test_selects_latest_attempt_or_explicit_attempt(self):
        names = "\n".join(f"reproduce-info-123-{n}-dtk2610-py3.12-torch2.13.0" for n in (1, 2))
        for attempt, selected in ((None, 2), (1, 1)):
            args = argparse.Namespace(repo="HYGON-AI/flash-train", run_id="123", torch_version="2.13.0", attempt=attempt)
            with patch.object(reproduce, "output", return_value=names), patch.object(reproduce, "run") as run:
                reproduce.download_manifest(args, Path("download"))
                self.assertIn(f"reproduce-info-123-{selected}-dtk2610-py3.12-torch2.13.0", run.call_args.args[0])

    def test_failed_build_inventory_excludes_the_pr_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest()))
            packages = [SimpleNamespace(metadata={"Name": "flash_train"}, version="0.1.0+pr123"),
                        SimpleNamespace(metadata={"Name": "torch"}, version="2.13.0+dtk2610.example")]
            env = {"CI_REPRO_DIR": directory, "CI_SETUP_OUTCOME": "success", "CI_BUILD_OUTCOME": "failure"}
            with patch.dict(os.environ, env, clear=True), patch.object(capture.sys, "argv", ["capture", "final"]), \
                    patch.object(capture.importlib.metadata, "distributions", return_value=packages), \
                    redirect_stdout(io.StringIO()):
                capture.main()
            data = json.loads(path.read_text())
            self.assertTrue(data["setup_complete"])
            self.assertEqual(data["failed_stage"], "build")
            self.assertEqual(data["project_package"], "0.1.0+pr123")
            self.assertNotIn("flash-train", data["packages"])


class SourceTests(unittest.TestCase):
    def test_remerges_recorded_commits_and_checks_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "origin"
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                               stderr=subprocess.DEVNULL).strip()
            git("init")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@local")
            git("config", "commit.gpgsign", "false")
            (repo / "base").write_text("base")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD")
            (repo / "head").write_text("PR change")
            git("add", ".")
            git("commit", "-m", "head")
            head = git("rev-parse", "HEAD")
            tree = git("rev-parse", "HEAD^{tree}")
            data = {"source": {"base_url": str(repo), "base_sha": base,
                               "head_url": str(repo), "head_sha": head, "tree": tree}}
            with redirect_stdout(io.StringIO()):
                reproduce.restore_source(data, root / "restored")
            self.assertEqual((root / "restored/head").read_text(), "PR change")
            data["source"]["tree"] = "0" * 40
            with self.assertRaisesRegex(RuntimeError, "Source tree differs"), redirect_stdout(io.StringIO()):
                reproduce.restore_source(data, root / "mismatch")


class DependencyTests(unittest.TestCase):
    def test_changed_aicc_binary_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = manifest()
            data["env"]["PYTHON_VERSION"] = f"{reproduce.sys.version_info.major}.{reproduce.sys.version_info.minor}"
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data))
            with patch.dict(os.environ, RUNNER_TEMP=directory), \
                    patch.object(reproduce, "urlopen", return_value=io.BytesIO(b"different nightly")), \
                    patch.object(reproduce, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    reproduce.inside_container(path)
                run.assert_not_called()

    def test_pins_torch_vendor_build_and_restores_other_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            data = manifest()
            data["env"]["PYTHON_VERSION"] = f"{reproduce.sys.version_info.major}.{reproduce.sys.version_info.minor}"
            data["aicc"]["sha256"] = hashlib.sha256(b"installer").hexdigest()
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(json.dumps(data))
            installed = [SimpleNamespace(metadata={"Name": name}, version=version)
                         for name, version in data["packages"].items()]
            real_write = Path.write_text
            def write(path, text, *args, **kwargs):
                if path.as_posix() == "/opt/dtk/hcu_llvm.info":
                    path = root / "written-hcu.info"
                return real_write(path, text, *args, **kwargs)
            yes = SimpleNamespace(stdout=io.BytesIO(), poll=lambda: 0, wait=lambda: 0)
            env = {"RUNNER_TEMP": directory, "PYPI_URL": "http://pypi.example:9929"}
            with patch.dict(os.environ, env), patch.object(Path, "write_text", write), \
                    patch.object(reproduce, "urlopen", return_value=io.BytesIO(b"installer")), \
                    patch.object(reproduce.subprocess, "Popen", return_value=yes), \
                    patch.object(reproduce.importlib.metadata, "distributions", side_effect=[[], installed]), \
                    patch.object(reproduce, "run") as run, redirect_stdout(io.StringIO()):
                reproduce.inside_container(path)
            commands = [call.args[0] for call in run.call_args_list]
            torch = next(command for command in commands if "torch==2.13.0+dtk2610.example" in command)
            self.assertIn("http://pypi.example:9929/das_nightly/dtk2610/+simple/", torch)
            self.assertIn("--no-deps", torch)
            self.assertEqual((root / "requirements-reproduce.txt").read_text(), "numpy==1.26.4\n")


class LifecycleTests(unittest.TestCase):
    def deploy(self, directory, step="build", fail=False, complete=True):
        root = Path(directory)
        path = root / "manifest.json"
        data = manifest()
        data["setup_complete"] = complete
        path.write_text(json.dumps(data))
        args = argparse.Namespace(manifest=str(path), repo=data["repository"], run_id="123",
                                  torch_version="2.13.0", attempt=None, step=step, work_dir=str(root / "work"))
        calls = []
        real_exists = Path.exists
        def exists(path):
            return True if path.as_posix().startswith(("/dev/", "/opt/hyhal")) else real_exists(path)
        def run(args, **kwargs):
            calls.append(list(map(str, args)))
        def logged(args, log):
            calls.append(args)
            if fail:
                raise RuntimeError("simulated installation failure")
        with ExitStack() as stack:
            stack.enter_context(patch.object(reproduce.sys, "platform", "linux"))
            stack.enter_context(patch.object(reproduce.shutil, "which", return_value="tool"))
            stack.enter_context(patch.object(reproduce, "run", side_effect=run))
            stack.enter_context(patch.object(reproduce, "execute_logged", side_effect=logged))
            stack.enter_context(patch.object(reproduce, "restore_source", side_effect=lambda d, p: p.mkdir()))
            stack.enter_context(patch.object(Path, "exists", exists))
            stack.enter_context(patch.object(reproduce.subprocess, "run", side_effect=run))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if fail or (step == "build" and not complete):
                with self.assertRaises(RuntimeError):
                    reproduce.deploy(args)
            else:
                reproduce.deploy(args)
        return calls, root / "work"

    def test_build_and_interactive_containers_are_stopped_but_not_deleted(self):
        for step in ("build", "test", None):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                calls, root = self.deploy(directory, step)
                create = next(call for call in calls if call[:2] == ["docker", "create"])
                self.assertNotIn("--rm", create)
                self.assertNotIn("/var/run/docker.sock", " ".join(create))
                self.assertEqual(calls[-1][:2], ["docker", "stop"])
                self.assertFalse(any(call[:2] == ["docker", "rm"] for call in calls))
                self.assertTrue((root / "enter.sh").is_file())
                self.assertTrue((root / "container-name.txt").is_file())
                self.assertNotIn("NIGHTLY", str(calls))

    def test_setup_failure_still_retains_container_and_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, root = self.deploy(directory, fail=True)
            self.assertEqual(calls[-1][:2], ["docker", "stop"])
            self.assertTrue((root / "enter.sh").is_file())

    def test_partial_snapshot_cannot_silently_be_used_for_build(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, root = self.deploy(directory, complete=False)
            self.assertFalse(any(call[:2] == ["docker", "create"] for call in calls))
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
