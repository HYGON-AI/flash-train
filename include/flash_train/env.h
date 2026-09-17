// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_ENV_H_
#define FTRAIN_ENV_H_

/**
 * @file env.h
 *
 * The single place declaring every environment variable flash-train reads
 * at runtime, paralleling control.h's build-time knobs. Each macro expands
 * to its variable's name. Name-list variables hold comma-separated names
 * whose entries are whitespace-trimmed; switch variables count as set when
 * their trimmed value is neither empty nor "0". An unset variable selects
 * the default behavior.
 */

/**
 * @brief Comma-separated allowlist of Primitive names.
 *
 * When set, a selection never returns a Primitive whose name is absent
 * from the list. Read once, at first use.
 */
#define FTRAIN_ENABLED_PRIMITIVES "FTRAIN_ENABLED_PRIMITIVES"

/**
 * @brief Comma-separated denylist of Primitive names.
 *
 * When set, a selection never returns a Primitive whose name is on the
 * list. Read once, at first use.
 */
#define FTRAIN_DISABLED_PRIMITIVES "FTRAIN_DISABLED_PRIMITIVES"

/**
 * @brief Comma-separated denylist of Finder names.
 *
 * When set, a Finder whose name is on the list contributes nothing to
 * selections. Read once, at first use.
 */
#define FTRAIN_DISABLED_FINDERS "FTRAIN_DISABLED_FINDERS"

/**
 * @brief Disables the selection cache.
 *
 * When set, every plan creation re-runs the Finders instead of reusing a
 * cached selection. Read once, at first use.
 */
#define FTRAIN_DISABLE_CACHE "FTRAIN_DISABLE_CACHE"

#endif
