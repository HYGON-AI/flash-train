// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FLASH_TRAIN_H_
#define FTRAIN_FLASH_TRAIN_H_

#include <flash_train/common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Executes one GEMM with the recommended Primitive.
 *
 * Convenience wrapper over the staged API: it binds the six Tensor ports
 * and the Gemm attributes internally, creates the Plan through the
 * library's selection cache, and executes primitive 0 -- the recommended
 * default -- on stream. It is equivalent to building one Gemm Pattern,
 * creating Ops, filling Args, creating a Plan, and executing its first
 * Primitive.
 *
 * All staged-API rules apply unchanged: a, b, c, and d are rank-two device
 * Tensors with consistent shapes (c's memory may be null, which drops the
 * beta * c term); alpha and beta are host scalar Tensors; execution is
 * asynchronous on stream, and the referenced memory and workspace must
 * stay valid until the enqueued work completes. The workspace must be at
 * least the recommended Primitive's requirement -- null and zero are
 * accepted when it requires none.
 *
 * @param a Device m-by-k input Tensor.
 * @param b Device k-by-n input Tensor.
 * @param c Device m-by-n input Tensor; its memory may be null.
 * @param d Device m-by-n output Tensor.
 * @param alpha Host scalar Tensor scaling a @ b.
 * @param beta Host scalar Tensor scaling c.
 * @param compute_type Numeric type used for computation.
 * @param workspace Workspace address.
 * @param workspace_bytes Size of workspace in bytes.
 * @param stream Stream used for execution.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when a parameter is inconsistent
 * or the workspace is insufficient; FTRAIN_STATUS_UNSUPPORTED when no
 * implementation satisfies the parameters; otherwise, the status of the
 * operation.
 */
FTRAIN_API FTrainStatus ftrainGemm(FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                                   FTrainStorageView alpha, FTrainStorageView beta, FTrainNumericType compute_type,
                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/**
 * @brief Selects and asynchronously executes sparse-attention backward.
 *
 * Ports and numerical semantics follow ftrainPatternAddSparseAttnBwd.
 * This is equivalent to assembling Args, selecting a Plan and executing
 * primitive 0. The initial implementation supports BF16 compute/storage
 * only, with FP32 lse/sink/dsink/scale and INT32 indices/lengths.
 *
 * Query the default candidate's workspace with a matching Plan using
 * ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan,0,...). Supply at least
 * that many device bytes; nonzero requirements need non-null workspace.
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
 * OUT_OF_MEMORY for a host allocation failure while building Args/Plan;
 * INTERNAL_ERROR for an immediate HIP initialization/launch failure or
 * unexpected host failure. No precedence is promised for multiple faults.
 * ftrainGetLastStatus/ftrainGetLastMessage preserve the returned failure
 * after internal cleanup; a successful call clears the thread-local error.
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
