#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Deploy a recorded flash-train build environment in a manually managed container."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


def run(args, **kwargs):
    print("+ " + shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def output(args, **kwargs):
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()


def public_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Expected an HTTP(S) URL without credentials, query or fragment")
    return value


def validate(data):
    if data.get("schema") != 1:
        raise ValueError("Unsupported reproduction manifest schema")
    for key in ("run_id", "attempt"):
        if not re.fullmatch(r"[0-9]+", data[key]):
            raise ValueError(f"Invalid {key}")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", data["repository"]):
        raise ValueError("Invalid repository")
    source = data["source"]
    for key in ("sha", "tree", "base_sha", "head_sha"):
        if key in source and not re.fullmatch(r"[0-9a-f]{40}", source[key]):
            raise ValueError(f"Invalid source {key}")
    for key in ("base_url", "head_url"):
        if key in source and not public_url(source[key]).startswith("https://github.com/"):
            raise ValueError("Only GitHub HTTPS source remotes are supported")
    image = data.get("image", {}).get("digest", "")
    if not image or not re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("No immutable image digest was captured; cannot reproduce from a mutable tag")
    for key in ("GITHUB_WORKSPACE", "RUNNER_TEMP"):
        path = PurePosixPath(data["env"][key])
        if not path.is_absolute() or ".." in path.parts or len(path.parts) < 3 or "," in str(path):
            raise ValueError(f"Unsafe recorded mount path: {key}")
    workspace = PurePosixPath(data["env"]["GITHUB_WORKSPACE"])
    temp = PurePosixPath(data["env"]["RUNNER_TEMP"])
    if workspace == temp or workspace in temp.parents or temp in workspace.parents:
        raise ValueError("Recorded workspace and temporary mounts overlap")
    for key in ("PYPI_URL", "AICC_INSTALLER_URL", "RESOURCE_SERVER_URL"):
        if data["env"].get(key):
            public_url(data["env"][key])
    if not data.get("aicc"):
        raise ValueError("No exact AICC package was captured (setup may have failed before package selection)")
    public_url(data["aicc"]["url"])
    if not re.fullmatch(r"[0-9a-f]{64}", data["aicc"]["sha256"]):
        raise ValueError("Missing AICC package checksum")
    if not data.get("packages"):
        raise ValueError("No Python package inventory was captured")
    if "flash-train" in data["packages"]:
        raise ValueError("The project wheel must be rebuilt from source, not restored from the index")
    for name, version in data["packages"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not re.fullmatch(r"[A-Za-z0-9!+._-]+", version):
            raise ValueError(f"Cannot pin package {name!r}")


def restore_source(data, directory):
    directory.mkdir()
    git = ["git", "-C", directory]
    run([*git, "init"])
    run([*git, "config", "core.autocrlf", "false"])
    run([*git, "remote", "add", "origin", data["source"]["base_url"]])
    run([*git, "fetch", "origin", data["source"]["base_sha"]])
    run([*git, "checkout", "--detach", "FETCH_HEAD"])
    if "head_sha" in data["source"]:
        run([*git, "remote", "add", "pr-head", data["source"]["head_url"]])
        run([*git, "fetch", "pr-head", data["source"]["head_sha"]])
        run([*git, "-c", "user.name=CI reproduction", "-c", "user.email=ci-reproduce@local",
             "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
             "merge", "--no-ff", "--no-edit", data["source"]["head_sha"]])
    tree = output([*git, "rev-parse", "HEAD^{tree}"])
    if tree != data["source"]["tree"]:
        raise RuntimeError(f"Source tree differs: expected {data['source']['tree']}, got {tree}. "
                           "Check Git merge versions/attributes; deployment stopped.")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def inside_container(manifest):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    validate(data)
    expected_python = data["env"]["PYTHON_VERSION"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected_python:
        raise RuntimeError(f"Expected Python {expected_python}; got {sys.version}")
    package = Path(os.environ["RUNNER_TEMP"]) / "reproduce-aicc.run"
    digest = hashlib.sha256()
    with urlopen(data["aicc"]["url"], timeout=120) as response, package.open("wb") as stream:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
            stream.write(block)
    if digest.hexdigest() != data["aicc"]["sha256"]:
        raise RuntimeError("AICC checksum mismatch; refusing to install a different nightly package")
    # Same installer invocation as hcu_llvm_installer.sh, using the pinned file.
    yes = subprocess.Popen(["yes"], stdout=subprocess.PIPE)
    try:
        run(["bash", package, "--dtk_dir", "/opt/dtk"], stdin=yes.stdout)
    finally:
        yes.stdout.close()
        if yes.poll() is None:
            yes.terminate()
        yes.wait()
    info = Path(manifest).with_name("hcu_llvm.info")
    Path("/opt/dtk/hcu_llvm.info").write_text(
        info.read_text(encoding="utf-8") if info.is_file() else f"Reproduced AICC: {data['aicc']['version']}\n")
    packages = data["packages"]
    current = {normalize(d.metadata["Name"]): d.version
               for d in importlib.metadata.distributions() if d.metadata["Name"]}
    changed = {name: version for name, version in packages.items()
               if current.get(normalize(name)) != version}
    dtk = data["env"]["DTK_VER"]
    index = os.environ["PYPI_URL"].rstrip("/")
    env = dict(os.environ, PIP_INDEX_URL=index + "/nightly/dtk2604/+simple/",
               PIP_TRUSTED_HOST=urlsplit(index).hostname)
    # The Torch vendor build must come from its DTK-specific index.
    torch_version = changed.pop("torch", None)
    if torch_version:
        torch_index = index + ("/das_nightly/dtk2610/+simple/" if dtk == "2610"
                               else f"/nightly/dtk{dtk}/+simple/")
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--index-url", torch_index,
             f"torch=={torch_version}"], env=env)
    if changed:
        lock = package.with_name("requirements-reproduce.txt")
        lock.write_text("".join(f"{name}=={version}\n" for name, version in changed.items()))
        run([sys.executable, "-m", "pip", "install", "--no-deps", "-r", lock], env=env)
    installed = {normalize(d.metadata["Name"]): d.version
                 for d in importlib.metadata.distributions() if d.metadata["Name"]}
    mismatches = [name for name, version in packages.items() if installed.get(normalize(name)) != version]
    if mismatches:
        raise RuntimeError(f"Package versions do not match: {mismatches}")
    print("Recorded AICC checksum and Python package versions verified.")


def download_manifest(args, destination):
    names = output(["gh", "api", "--paginate", f"repos/{args.repo}/actions/runs/{args.run_id}/artifacts",
                    "--jq", ".artifacts[] | select(.expired == false) | .name"]).splitlines()
    pattern = re.compile(rf"reproduce-info-{args.run_id}-(\d+)-dtk\d+-py[\d.]+-torch{re.escape(args.torch_version)}")
    matches = [(int(match[1]), name) for name in names if (match := pattern.fullmatch(name))]
    if args.attempt:
        matches = [pair for pair in matches if pair[0] == args.attempt]
    if not matches:
        raise RuntimeError("No reproduction artifact found. Historical runs before this feature, "
                           "expired artifacts, or failures before checkout/metadata capture cannot be replayed.")
    latest = max(attempt for attempt, _ in matches)
    selected = [name for attempt, name in matches if attempt == latest]
    if len(selected) != 1:
        raise RuntimeError("Multiple matching artifacts; select a run with a unique Torch matrix entry")
    run(["gh", "run", "download", args.run_id, "--repo", args.repo,
         "--name", selected[0], "--dir", destination])
    return destination / "manifest.json"


def deploy(args):
    if sys.platform != "linux":
        raise RuntimeError("Run on a Linux HCU node with Docker, Git, Python 3.10+ and GitHub CLI")
    for command in ("git", "docker", *( [] if args.manifest else ["gh"] )):
        if not shutil.which(command):
            raise RuntimeError(f"Required command is missing: {command}")
    run(["docker", "info"], stdout=subprocess.DEVNULL)
    for device in ("/dev/kfd", "/dev/dri", "/opt/hyhal"):
        if not Path(device).exists():
            raise RuntimeError(f"HCU node prerequisite is missing: {device}")
    with tempfile.TemporaryDirectory(prefix="ft-repro-info-") as temporary:
        manifest = Path(args.manifest).resolve() if args.manifest else download_manifest(args, Path(temporary))
        data = json.loads(manifest.read_text(encoding="utf-8"))
        validate(data)
        if (data["repository"] != args.repo or data["run_id"] != args.run_id
                or data["env"]["TORCH_VERSION"] != args.torch_version
                or (args.attempt and int(data["attempt"]) != args.attempt)):
            raise RuntimeError("Manifest does not match the requested repository/run/attempt/Torch")
        if args.step in ("build", "test") and not data["setup_complete"]:
            raise RuntimeError("CI setup did not complete; use --step setup or omit --step to inspect "
                               "the recorded partial environment. It is not a ready-to-build snapshot.")
        if args.step == "test" and data["env"].get("CI_RUN_TESTS") != "true":
            raise RuntimeError("This push job did not run tests; use --step build to replay it")
        identifier = f"{data['run_id']}-{data['attempt']}-torch{args.torch_version.replace('.', '')}"
        root = Path(args.work_dir or Path.cwd() / ".ci-reproduce" / identifier).resolve()
        if "," in str(root):
            raise ValueError("Docker mount directory must not contain a comma")
        root.mkdir(parents=True, exist_ok=False)
        root.chmod(0o700)
        info = root / "info"
        info.mkdir()
        for name in ("manifest.json", "hcu_llvm.info"):
            if (manifest.parent / name).is_file():
                shutil.copy2(manifest.parent / name, info / name)
        # Pin the deploying helper as well, so later edits to this checkout do
        # not change how an existing reproduction container is entered.
        tools = root / "tools"
        tools.mkdir()
        shutil.copy2(Path(__file__).resolve(), tools / "reproduce-ci.py")
        (root / "tmp").mkdir()
        (root / "logs").mkdir()
        restore_source(data, root / "workspace")
        env = dict(data["env"])
        # Only capture allowlisted build variables; credentials never go here.
        env_file = root / "container.env"
        env_file.write_text("".join(f"{key}={value}\n" for key, value in env.items()
                                    if "\n" not in value and "\r" not in value))
        env_file.chmod(0o600)
        name = f"ft-repro-{identifier}-{os.urandom(3).hex()}"
        run(["docker", "pull", data["image"]["digest"]])
        run(["docker", "create", "--name", name, "--label", "flash-train.reproduce=true",
             "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
             "--shm-size=80G", "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
             "--privileged=true", "--env-file", env_file,
             "--mount", f"type=bind,src={root / 'workspace'},dst={env['GITHUB_WORKSPACE']}",
             "--mount", f"type=bind,src={root / 'tmp'},dst={env['RUNNER_TEMP']}",
             "--mount", f"type=bind,src={info},dst=/ci-reproduce-info,readonly",
             "--mount", f"type=bind,src={tools},dst=/ci-reproduce-tools,readonly",
             "--mount", "type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly",
             "--workdir", env["GITHUB_WORKSPACE"], "--entrypoint", "tail",
             data["image"]["digest"], "-f", "/dev/null"])
        (root / "container-name.txt").write_text(name + "\n")
        entry = "source .github/scripts/common.sh; exec bash --noprofile --norc"
        (root / "enter.sh").write_text("#!/usr/bin/env bash\nset -e\n"
            + shlex.join(["docker", "start", name]) + " >/dev/null\n"
            + shlex.join(["docker", "exec", "-it", name, "bash", "-c", entry]) + "\n")
        print(f"\nContainer: {name}\nFiles: {root}\nRe-enter: bash {shlex.quote(str(root / 'enter.sh'))}"
              f"\nStop: docker stop {name}\nDelete manually: docker rm {name}\n", flush=True)
        try:
            run(["docker", "start", name])
            setup = "source /opt/dtk/env.sh; exec python3 /ci-reproduce-tools/reproduce-ci.py --inside-container /ci-reproduce-info/manifest.json"
            with (root / "logs" / "setup.log").open("w") as log:
                execute_logged(["docker", "exec", name, "bash", "-eo", "pipefail", "-c", setup], log)
            if args.step == "setup":
                return
            if args.step in ("build", "test"):
                command = "source .github/scripts/common.sh; args=(); "
                if env.get("CPP_TESTS") == "true" and env.get("CI_RUN_TESTS") == "true":
                    command += "args+=(--build-tests); "
                command += ('bash .github/scripts/_build_hcu_wheel.sh --out-dir "$CI_WORK_DIR/dist" "${args[@]}"; '
                            'CIUpload REPAIR --dtk_version "$DTK_VER" --torch_version "$TORCH_VERSION" '
                            '--git_commit "' + data["source"]["sha"][:6] + '" --outputdir "$WHEEL_OUTPUT" '
                            '-f "$CI_WORK_DIR"/dist/*.whl')
                if args.step == "test":
                    command += '; bash .github/scripts/test-wheel.sh'
                with (root / "logs" / f"{args.step}.log").open("w") as log:
                    execute_logged(["docker", "exec", name, "bash", "-eo", "pipefail", "-c", command], log)
            else:
                run(["docker", "exec", "-it", name, "bash", "-eo", "pipefail", "-c", entry])
        finally:
            # Leave the writable layer and all host files intact. Never --rm.
            subprocess.run(["docker", "stop", name], check=False)


def execute_logged(command, log):
    print("+ " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    finally:
        process.stdout.close()
    if process.wait():
        raise RuntimeError(f"Container command failed with exit code {process.returncode}; see local logs")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--inside-container":
        inside_container(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub owner/repository")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--torch-version", required=True, help="Matrix version, e.g. 2.13.0")
    parser.add_argument("--attempt", type=int, help="Default: latest available attempt for this Torch version")
    parser.add_argument("--step", choices=("setup", "build", "test"),
                        help="test builds the wheel first; omit to enter an interactive shell")
    parser.add_argument("--work-dir", help="New directory for source, logs and metadata; must not already exist")
    parser.add_argument("--manifest", help="Use an already downloaded manifest.json instead of GitHub CLI")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or not args.run_id.isdigit():
        parser.error("Invalid repository or run ID")
    deploy(args)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
