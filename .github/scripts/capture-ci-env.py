#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Capture build reproduction metadata without exporting CI credentials."""
import argparse
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ENV_KEYS = (
    "DTK_VER", "PYTHON_VERSION", "NUMPY_VERSION", "TORCH_VERSION", "CPP_TESTS",
    "MAX_JOBS", "PYTHON", "PYPI_URL", "AICC_INSTALLER_URL", "RESOURCE_SERVER_URL",
    "GITHUB_WORKSPACE", "RUNNER_TEMP", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
    "GITHUB_JOB", "GITHUB_SHA", "GITHUB_REPOSITORY", "CI_RUN_TESTS", "DTK_ROOT", "FTRAIN_AICC",
)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def public_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("https", "http") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Reproduction URLs must be HTTP(S) without credentials, query or fragment")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docker_get(path):
    # Job containers mount the host Docker socket, but need not contain its CLI.
    connection = http.client.HTTPConnection("localhost", timeout=30)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.settimeout(30)
    connection.sock.connect("/var/run/docker.sock")
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Docker inspect returned {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


def capture_initial():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    pr = event.get("pull_request")
    commit = git("rev-parse", "HEAD")
    source = {"sha": commit, "tree": git("rev-parse", "HEAD^{tree}"),
              "base_url": public_url(f"https://github.com/{os.environ['GITHUB_REPOSITORY']}.git")}
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and pr:
        parents = git("show", "-s", "--format=%P", "HEAD").split()
        if len(parents) != 2 or parents[1] != pr["head"]["sha"]:
            raise RuntimeError("Checkout is not the expected PR merge commit")
        source.update(base_sha=parents[0], head_sha=parents[1],
                      head_url=public_url(pr["head"]["repo"]["clone_url"]))
    else:
        source["base_sha"] = commit
    env = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
    for key in ("PYPI_URL", "AICC_INSTALLER_URL", "RESOURCE_SERVER_URL"):
        if env.get(key):
            public_url(env[key])
    data = {"schema": 1, "repository": os.environ["GITHUB_REPOSITORY"],
            "run_id": os.environ["GITHUB_RUN_ID"], "attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "runner": os.environ.get("RUNNER_NAME"), "source": source, "env": env,
            "setup_complete": False}
    try:
        container = docker_get(f"/containers/{os.environ['CI_CONTAINER_ID']}/json")
        image = docker_get(f"/images/{container['Image']}/json")
        # Keep an immutable registry reference, not the mutable tag.
        data["image"] = {"id": container["Image"], "digest": (image.get("RepoDigests") or [None])[0],
                         "tag": container["Config"]["Image"]}
    except Exception as exc:
        data["image_error"] = str(exc)
    return data


def capture_aicc(data, selection, package):
    text = Path(selection).read_text(encoding="utf-8")
    urls = re.findall(r"https?://\S+", text)
    url = next((url for url in urls if urlsplit(url).path.endswith(".run")), None)
    if not url:
        raise RuntimeError("AICC selection output does not contain a .run URL")
    public_url(url)
    match = re.search(r"^完整版本:\s*(\S+)\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("AICC selection output does not contain the exact version")
    if Path(package).name != Path(unquote(urlsplit(url).path)).name:
        raise RuntimeError("AICC download filename does not match the selected URL")
    data["aicc"] = {"url": url, "version": match[1], "sha256": sha256(package)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initial", "aicc", "final"))
    parser.add_argument("--selection")
    parser.add_argument("--package")
    args = parser.parse_args()
    directory = Path(os.environ["CI_REPRO_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    data = capture_initial() if args.stage == "initial" else json.loads(path.read_text(encoding="utf-8"))
    if args.stage == "aicc":
        capture_aicc(data, args.selection, args.package)
    elif args.stage == "final":
        # Record versions even when setup failed halfway through; do not claim
        # such a snapshot is a fully prepared build environment.
        packages = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                    for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
        # The wheel under test may have been installed from this PR. It is not
        # a dependency available on the package index: rebuild it from source.
        data["project_package"] = packages.pop("flash-train", None)
        data["packages"] = dict(sorted(packages.items()))
        data["setup_complete"] = os.environ.get("CI_SETUP_OUTCOME") == "success"
        data["outcomes"] = {key: os.environ.get(f"CI_{key.upper()}_OUTCOME", "skipped")
                            for key in ("setup", "build", "test")}
        for name in ("setup", "build", "test"):
            if data["outcomes"][name] == "failure":
                data["failed_stage"] = name
                break
        info = Path("/opt/dtk/hcu_llvm.info")
        if info.is_file():
            (directory / "hcu_llvm.info").write_text(info.read_text(encoding="utf-8"))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(f"Reproduction manifest: {path}")
    if args.stage == "final" and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"\nReproduction metadata: run `{data['run_id']}`, attempt `{data['attempt']}`, "
                          f"Torch `{data['env']['TORCH_VERSION']}`. "
                          f"Prepared environment: `{data['setup_complete']}`.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: cannot capture reproduction metadata: {exc}", file=sys.stderr)
        sys.exit(1)
