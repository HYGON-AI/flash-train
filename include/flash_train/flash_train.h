// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FLASH_TRAIN_H_
#define FTRAIN_FLASH_TRAIN_H_

#include <flash_train/common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Selects and asynchronously executes sparse-attention backward.
 *
 * Ports and numerical semantics follow ftrainSparseAttnBwdCreate. This is
 * equivalent to creating a Plan with ftrainSparseAttnBwdCreate() and
 * executing its primitive 0. Only BF16 compute/storage is supported, with
 * FP32 lse/sink/dsink/scale and INT32 indices/lengths.
 *
 * Query the default candidate's workspace with a Plan from
 * ftrainSparseAttnBwdCreate() and
 * ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(plan,0,...). Supply at
 * least that many device bytes; nonzero requirements need non-null workspace.
 * The current baseline needs FP32-aligned workspace. It initializes its
 * scratch and outputs on stream every call. Device buffers/workspace must
 * remain valid until stream completion. Descriptor arrays and the host
 * scale can be released/reused on return; their metadata/value are copied.
 * SUCCESS means work was enqueued, without synchronizing. Later device
 * execution faults are reported by HIP when the caller synchronizes.
 *
 * @return SUCCESS if selected and enqueued; INVALID_ARGUMENT for invalid
 * view enums/dimensions/metadata, wrong rank/shape/placement, null host
 * scalar, invalid compute enum, S==0, K==0, or insufficient/null workspace;
 * OVERFLOW if a tensor's byte extent cannot be represented in uint64_t;
 * UNSUPPORTED for structurally valid ports with no applicable implementation
 * (including null required device memory, unsupported precision, layout,
 * alignment, D/V, empty T/H, INT32 indexing limits, architecture or LDS);
 * OUT_OF_MEMORY for a host allocation failure while selecting a Plan;
 * INTERNAL_ERROR for an immediate HIP initialization/launch failure or
 * unexpected host failure. No precedence is promised for multiple faults.
 * A successful call clears the thread-local error; a failing one leaves its
 * status and message there.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView out,
                                            FTrainStorageView dout, FTrainStorageView lse, FTrainStorageView attn_sink,
                                            FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                            FTrainStorageView dq, FTrainStorageView dkv, FTrainStorageView dsink,
                                            FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream);

#ifdef __cplusplus
}
#endif

#endif
