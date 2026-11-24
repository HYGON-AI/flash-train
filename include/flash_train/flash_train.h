// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FLASH_TRAIN_H_
#define FTRAIN_FLASH_TRAIN_H_

#include <flash_train/common.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque resource handles. */

typedef struct FTrainSparseAttnBwdPlanStruct* FTrainSparseAttnBwdPlan;

/* sparse_attn_bwd (BF16). */

/**
 * @brief Selects implementations for sparse-attention backward and returns
 * them as a reusable Plan bound to the supplied storage.
 *
 * Let T=query tokens, S=KV tokens, H=query heads, D=Q/K width, V=value
 * width, and K=selection width. Port contracts:
 * - q/dq: BF16 (T,H,D); kv/dkv: BF16 (S,D) or (S,1,D), with identical
 *   shapes within each input/gradient pair.
 * - out/dout: BF16 (T,H,V), identical shapes. out is the sink-inclusive
 *   forward output; dout is its incoming gradient.
 * - lse: FP32 (T,H), the KV-only logsumexp, excluding the sink logit.
 * - attn_sink/dsink: FP32 (H), sink logits and their reduced gradients.
 * - topk_indices: INT32 (T,K), global row indices in the shared KV tensor.
 *   Indices outside [0,S) are masked; duplicates accumulate gradients.
 * - topk_length: INT32 (T). Null memory consumes all K entries; otherwise
 *   each row consumes clamp(topk_length[t],0,K) entries. Even when memory is
 *   null, this port must describe an INT32 device tensor of shape (T).
 * - softmax_scale: non-null rank-zero FP32 host tensor, copied by value
 *   when the Plan is configured. Other ports describe device memory.
 *
 * All ports use FTRAIN_INDEX_TYPE_CONTINUOUS with null strides. Required
 * data memory must be non-null; q/kv/out/dout/dq/dkv require 16-byte
 * alignment, and the other ports require their element-type alignment.
 * Current BF16 support: gfx936/gfx938; T,H,K>0; S>0; D=512 or 576;
 * V=D or 512; each tensor's dimensions and element count fit INT32;
 * dynamic shared-memory use must fit the selected device. Unsupported
 * shapes/types/layouts/alignment/null data return UNSUPPORTED on selection.
 *
 * For valid selected keys i, p_i=exp(scale*dot(q,kv_i))/Z, with
 * Z=exp(lse)+exp(attn_sink). With delta=dot(out,dout), gradients are
 * dq=scale*sum_i p_i*(dot(dout,kv_i[:V])-delta)*kv_i,
 * dkv_i[:V]+=p_i*dout, dkv_i+=scale*p_i*(dot(dout,kv_i[:V])-delta)*q,
 * and dsink[h]=sum_t -exp(attn_sink[h])/Z*delta. Outputs overwrite all
 * prior values; inputs are read-only. Buffers must not overlap, and their
 * allocations/devices must be valid for the selected device/stream. These
 * storage/content preconditions are the caller's responsibility; device
 * values, buffer capacity, and overlap are not scanned on the host.
 *
 * The returned Plan is bound to the calling thread's current device and to
 * the supplied storage. Selections are keyed by port metadata, never by
 * address: raw memory addresses do not prevent a compatible selection from
 * being reused.
 *
 * The returned Plan holds the selected Primitives ordered best-first;
 * primitive 0 is the recommended default. When a Finder contributes a
 * ranked candidate, its candidates are the selection; when none does, the
 * Plan holds every applicable registered Primitive in registration order.
 * Workspace budgets are the caller's concern: walk the Primitives with
 * ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes() and keep the first
 * index whose requirement fits the caller's workspace; execution enforces
 * each Primitive's requirement.
 *
 * A successful call always returns a Plan holding at least one usable
 * Primitive; ftrainSparseAttnBwdGetNumPrimitives() is not an availability
 * check. The library copies every port's metadata and the host scale value;
 * the caller may release those arrays and the scale storage on return. The
 * described device memory is neither copied nor owned and must stay valid
 * until every execution on the returned Plan completes.
 *
 * @param plan Non-null output pointer. On success, receives the created Plan.
 * It is unchanged on failure.
 * @param q Device query Tensor (T,H,D).
 * @param kv Device shared KV Tensor (S,D) or (S,1,D).
 * @param out Device sink-inclusive forward output (T,H,V).
 * @param dout Device incoming gradient of out (T,H,V).
 * @param lse Device KV-only logsumexp (T,H).
 * @param attn_sink Device sink logits (H).
 * @param topk_indices Device selected KV row indices (T,K).
 * @param topk_length Device per-token valid entry counts (T); memory may be
 * null.
 * @param dq Device output gradient of q (T,H,D).
 * @param dkv Device output gradient of kv, shaped like kv.
 * @param dsink Device output reduced gradient of attn_sink (H).
 * @param softmax_scale Host scalar Tensor.
 * @param compute_type Numeric type used for computation.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when plan is null, compute_type is
 * INVALID, COUNT, or out of range, a port view has invalid
 * enums/dimensions/metadata, a port has the wrong rank/shape/placement, or
 * the host scalar is null; FTRAIN_STATUS_OVERFLOW when a tensor's byte extent
 * cannot be represented in uint64_t; FTRAIN_STATUS_UNSUPPORTED when no
 * available implementation satisfies the parameters; otherwise, the status of
 * the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdCreate(FTrainSparseAttnBwdPlan* plan, FTrainStorageView q,
                                                  FTrainStorageView kv, FTrainStorageView out, FTrainStorageView dout,
                                                  FTrainStorageView lse, FTrainStorageView attn_sink,
                                                  FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                                  FTrainStorageView dq, FTrainStorageView dkv, FTrainStorageView dsink,
                                                  FTrainStorageView softmax_scale, FTrainNumericType compute_type);

/**
 * @brief Destroys a sparse-attention backward Plan.
 *
 * @param plan Plan to destroy.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdDestroy(FTrainSparseAttnBwdPlan plan);

/**
 * @brief Returns the number of Primitives in a sparse-attention backward Plan.
 *
 * The count is at least 1 for every successfully created Plan and exceeds
 * 1 when several implementations apply to the Plan's parameters; the
 * Primitives are ordered best-first.
 *
 * @param plan Plan to query.
 * @param num_primitives Non-null output pointer that receives the count.
 * It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdGetNumPrimitives(FTrainSparseAttnBwdPlan plan, uint64_t* num_primitives);

/**
 * @brief Returns the name of one Primitive in a sparse-attention backward Plan.
 *
 * The name is the Primitive's identity inside its engine: it is the
 * string the FTRAIN_ENABLED_PRIMITIVES and FTRAIN_DISABLED_PRIMITIVES
 * variables filter by. The returned pointer is owned by the Plan and
 * stays valid until the Plan is destroyed.
 *
 * @param plan Plan to query.
 * @param primitive_index Index of the Primitive within the Plan.
 * @param name Non-null output pointer that receives the name. It is
 * unchanged on failure.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when primitive_index is out of
 * range; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdGetPrimitiveName(FTrainSparseAttnBwdPlan plan, uint64_t primitive_index,
                                                            const char** name);

/**
 * @brief Returns the workspace one Primitive of a sparse-attention backward
 * Plan requires.
 *
 * @param plan Plan to query.
 * @param primitive_index Index of the Primitive within the Plan.
 * @param workspace_bytes Non-null output pointer that receives the required
 * size in bytes. It is unchanged on failure.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when primitive_index is out of
 * range; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(FTrainSparseAttnBwdPlan plan,
                                                                              uint64_t primitive_index,
                                                                              uint64_t* workspace_bytes);

/**
 * @brief Executes one Primitive of a sparse-attention backward Plan on a
 * stream.
 *
 * Every Primitive in the Plan is bound to the current device observed by
 * ftrainSparseAttnBwdCreate(). The supplied stream and referenced Tensor memory
 * must be compatible with that device.
 *
 * The workspace size must be at least the value returned by
 * ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes() for the same index. A
 * non-null workspace is required when that value is nonzero.
 *
 * This function does not synchronize stream. Any submitted work is
 * asynchronous. The Plan, every referenced Tensor memory region, and
 * workspace must remain valid until that work completes. The caller must
 * externally synchronize concurrent executions of the same Plan.
 *
 * @param plan Plan to execute.
 * @param primitive_index Index of the Primitive to execute; 0 runs the
 * recommended default.
 * @param workspace Workspace address.
 * @param workspace_bytes Size of workspace in bytes.
 * @param stream Stream used for execution.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when primitive_index is out of
 * range, the calling thread's device differs from the Plan's device, or the
 * workspace is insufficient; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdExecute(FTrainSparseAttnBwdPlan plan, uint64_t primitive_index,
                                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream);

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
