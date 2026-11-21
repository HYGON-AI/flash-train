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

/*
 * KDA. Training needs none of these: the defaults are the measured best policy, and the PyTorch
 * entry's `disable_recompute` argument is the caller-facing memory / speed choice (it overrides
 * the three save-policy variables below). They exist for A/B runs and debugging; none of them
 * changes a result bit.
 */

/**
 * @brief KDA (flash_train.torch): the forward saves the l2-normalized qn / kn in place of q / k
 * ("1", default) or not ("0"). Read at every forward call.
 */
#define FTRAIN_KDA_SAVE_QKN "FTRAIN_KDA_SAVE_QKN"

/**
 * @brief KDA (flash_train.torch): the forward saves the recurrent state h so the backward runs no
 * recurrence: "auto" (default, at most 64 heads per call), "1" (always) or "0" (never). Read at
 * every forward call.
 */
#define FTRAIN_KDA_SAVE_H "FTRAIN_KDA_SAVE_H"

/**
 * @brief KDA (flash_train.torch): the forward hands its full recurrent state to the backward when
 * h is not saved: "auto" (default, only in a reentrant activation-checkpoint recompute), "1"
 * (whenever a backward follows) or "0" (never). Read at every forward call.
 */
#define FTRAIN_KDA_KEEP_STATE "FTRAIN_KDA_KEEP_STATE"

/**
 * @brief KDA: switch enabling the G2 kernels' tuning knobs (KDA_G2_*, G2_*_STOP, inherited from
 * the hip_kda research tree). Unset, every knob takes its default whatever the environment holds.
 * Read once, at first use.
 */
#define FTRAIN_KDA_DEV "FTRAIN_KDA_DEV"

#endif
