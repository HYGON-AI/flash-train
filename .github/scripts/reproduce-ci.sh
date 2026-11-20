#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/reproduce-ci.py" "$@"
