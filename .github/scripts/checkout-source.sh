#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"
git config --global --add safe.directory "${GITHUB_WORKSPACE}"
source_sha="$(git rev-parse HEAD)"
expected_sha="${EXPECTED_SOURCE_SHA:-${GITHUB_SHA:?}}"
if [[ "$source_sha" != "$expected_sha" ]]; then
  echo "::error::Expected source $expected_sha but checked out $source_sha."
  exit 1
fi
echo "Source commit: $source_sha"

if [[ -n "${PR_BASE_SHA:-}" ]]; then
  git checkout --detach "$PR_BASE_SHA"
  # Keep the base first, so the build and local reproduction use the same merge direction.
  merge_date="$(git show -s --format=%cI "$source_sha")"
  GIT_AUTHOR_DATE="$merge_date" GIT_COMMITTER_DATE="$merge_date" \
    git -c user.name='flash-train CI' -c user.email='ci@localhost' \
    -c core.hooksPath=/dev/null -c commit.gpgsign=false \
    merge --no-ff -m "Merge PR head $source_sha into $PR_BASE_SHA" "$source_sha"
  echo "Integrated PR base: $PR_BASE_SHA"
fi
echo "Integrated commit: $(git rev-parse HEAD)"
git show -s --format='Parents: %P'
