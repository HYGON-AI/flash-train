#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Compatibility entry point; all builds use the same HCU environment checks.
exec bash "$(dirname "${BASH_SOURCE[0]}")/.github/scripts/_build_hcu_wheel.sh" "$@"
