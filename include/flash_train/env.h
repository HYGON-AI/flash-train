// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

/**
 * @brief BF16 sparse-attention backward workspace policy.
 *
 * Unset, empty or "0": full workspace (default, high performance).
 * "1": two query batches sharing scratch for the D=576, H=64,
 * T=4096/4097, T_kv=4096, topk=2048 compact-values path.
 * Other values return INVALID_ARGUMENT. Other precisions are unaffected.
 * Sampled at Problem construction and encoded in the selection key.
 * A configured Plan retains its policy even if the environment changes.
 */
#define DSA_BWD_WORKSPACE "DSA_BWD_WORKSPACE"

/** @brief Cached BF16 sparse-backward path: unset/nonzero enables, atoi(value)==0 disables.
 * Sampled at BF16 Problem construction and encoded in the selection key.
 * A configured Plan retains its policy. Other precisions are unaffected.
 */
#define DSA_SAB2_CACHED "DSA_SAB2_CACHED"

/** @brief Projection K-split override; unset/0 uses automatic selection.
 * Parsed with atoi and read once per precision implementation.
 */
#define MHX_KS "MHX_KS"

/** @brief Nonzero atoi(value) disables the projection/compute-H v2 fast path.
 * Unset defaults to zero. Read once per precision implementation.
 */
#define MHX_NOV2 "MHX_NOV2"

/** @brief Projection/compute-H v2 LDS layout override (atoi, nonzero enables).
 * Unset uses the compute-precision default. Read once per precision implementation.
 */
#define MHX_CFREE "MHX_CFREE"

#endif
