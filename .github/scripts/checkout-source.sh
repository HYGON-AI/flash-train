#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"
git config --global --add safe.directory "${GITHUB_WORKSPACE}"
# PRs use the same synthetic merge commit in every job, selected by GitHub.
# Push and manual runs test the exact event SHA.
test "$(git rev-parse HEAD)" = "${GITHUB_SHA:?}"
echo "Integrated commit: $(git rev-parse HEAD)"
git show -s --format='Parents: %P'
