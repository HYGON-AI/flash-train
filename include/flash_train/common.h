// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_COMMON_H_
#define FTRAIN_COMMON_H_

#include <stdbool.h>
#include <stdint.h>

#include <flash_train/control.h>
#include <flash_train/version.h>

#if FTRAIN_PLATFORM_HYGON_HIP
#    include <hip/hip_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if FTRAIN_PLATFORM_HYGON_HIP
typedef hipStream_t FTrainStream;
typedef int32_t FTrainDeviceId;
#endif

/* Enumeration types. */

typedef uint8_t FTrainStatus;
enum {
    FTRAIN_STATUS_SUCCESS          = 0,
    FTRAIN_STATUS_INVALID_ARGUMENT = 1,
    FTRAIN_STATUS_OUT_OF_MEMORY    = 2,
    FTRAIN_STATUS_OVERFLOW         = 3,
    FTRAIN_STATUS_UNSUPPORTED      = 4,
    FTRAIN_STATUS_INTERNAL_ERROR   = 50,
};

typedef uint8_t FTrainNumericType;
enum {
    FTRAIN_NUMERIC_TYPE_INVALID    = 0,
    FTRAIN_NUMERIC_TYPE_FP64       = 1,
    FTRAIN_NUMERIC_TYPE_INT64      = 2,
    FTRAIN_NUMERIC_TYPE_UINT64     = 3,
    FTRAIN_NUMERIC_TYPE_FP32       = 4,
    FTRAIN_NUMERIC_TYPE_INT32      = 5,
    FTRAIN_NUMERIC_TYPE_UINT32     = 6,
    FTRAIN_NUMERIC_TYPE_FP16       = 7,
    FTRAIN_NUMERIC_TYPE_BF16       = 8,
    FTRAIN_NUMERIC_TYPE_INT16      = 9,
    FTRAIN_NUMERIC_TYPE_UINT16     = 10,
    FTRAIN_NUMERIC_TYPE_FP8_E4M3FN = 11,
    FTRAIN_NUMERIC_TYPE_FP8_E5M2   = 12,
    FTRAIN_NUMERIC_TYPE_UFP8_E8M0  = 13,
    FTRAIN_NUMERIC_TYPE_UFP8_E5M3  = 14,
    FTRAIN_NUMERIC_TYPE_UFP8_E6M2  = 15,
    FTRAIN_NUMERIC_TYPE_INT8       = 16,
    FTRAIN_NUMERIC_TYPE_UINT8      = 17,
    FTRAIN_NUMERIC_TYPE_FP4_E2M1   = 18,
    FTRAIN_NUMERIC_TYPE_FP4_S1P2   = 19,
    FTRAIN_NUMERIC_TYPE_INT4       = 20,
    FTRAIN_NUMERIC_TYPE_UINT4      = 21,
    FTRAIN_NUMERIC_TYPE_UFP1_E1M0  = 22,
    FTRAIN_NUMERIC_TYPE_UINT1      = 23,
    FTRAIN_NUMERIC_TYPE_COUNT,
};

typedef uint8_t FTrainIndexType;
enum {
    FTRAIN_INDEX_TYPE_INVALID                  = 0,
    FTRAIN_INDEX_TYPE_CONTINUOUS               = 1,
    FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED = 2,
    FTRAIN_INDEX_TYPE_GROUPED_CONTINUOUS       = 3,
    FTRAIN_INDEX_TYPE_COUNT,
};

/* Storage descriptions. */

/**
 * @brief Describes concrete Tensor storage and layout.
 *
 * A zero-dimensional view represents a scalar containing one logical element,
 * and its memory address must be non-null. A non-scalar view may have a null
 * memory address. When passed to an Args setter, Flash Train copies the
 * dimension and stride metadata before returning but does not copy or own the
 * referenced memory.
 */
typedef struct {
    /** Address of the Tensor storage. */
    void* memory;

    /**
     * Array of num_dims non-negative dimension sizes. It must be non-null when
     * num_dims is nonzero.
     */
    const int64_t* dims;

    /**
     * Array of num_dims strides measured in logical elements. It must be
     * non-null when index_type is FTRAIN_INDEX_TYPE_INVALID and must be null
     * when index_type names a predefined layout.
     */
    const int64_t* strides;

    /** Number of dimensions. Zero dimensions describe one scalar element. */
    uint8_t num_dims;

    /** Numeric type of each logical element. INVALID and COUNT are not valid types. */
    FTrainNumericType numeric_type;

    /**
     * Predefined storage layout. FTRAIN_INDEX_TYPE_INVALID means that strides
     * supplies the layout; any other defined value means that strides is null.
     */
    FTrainIndexType index_type;

    /** True when memory references host storage; false when it references device storage. */
    bool is_host_memory;
} FTrainStorageView;

/**
 * @defgroup FTrainRoleIds Pattern role IDs
 *
 * All public operand and operation ID types are opaque role identifiers. Callers
 * may copy, store, and pass them back to Flash Train, but must not interpret or
 * modify their values. They own no resource and require no destruction.
 * @{
 */

// clang-format off

typedef struct { uint64_t opaque; } FTrainTensorId;
typedef struct { uint64_t opaque; } FTrainTensorListId;
typedef struct { uint64_t opaque; } FTrainGroupedTensorId;
typedef struct { uint64_t opaque; } FTrainGemmOpId;
typedef struct { uint64_t opaque; } FTrainGroupedABCDGemmOpId;
typedef struct { uint64_t opaque; } FTrainGroupedBCDGemmOpId;
typedef struct { uint64_t opaque; } FTrainGroupedABGemmOpId;
/* DSA operators; one ID type per operator family. */
typedef struct { uint64_t opaque; } FTrainIndexerFwdOpId;
typedef struct { uint64_t opaque; } FTrainIndexerFwdThdOpId;
typedef struct { uint64_t opaque; } FTrainDenseIndexerScoreOpId;
typedef struct { uint64_t opaque; } FTrainDenseIndexerScoreThdOpId;
typedef struct { uint64_t opaque; } FTrainDenseAttnScoreOpId;
typedef struct { uint64_t opaque; } FTrainDenseAttnScoreThdOpId;
typedef struct { uint64_t opaque; } FTrainSparseIndexerPredictOpId;
typedef struct { uint64_t opaque; } FTrainSparseAttnTargetOpId;
typedef struct { uint64_t opaque; } FTrainSparseAttnFwdOpId;
typedef struct { uint64_t opaque; } FTrainSparseAttnFwdIndexerOpId;
typedef struct { uint64_t opaque; } FTrainSparseAttnBwdOpId;
typedef struct { uint64_t opaque; } FTrainIndexerBwdSparseOpId;
typedef struct { uint64_t opaque; } FTrainIndexerBwdDenseOpId;
typedef struct { uint64_t opaque; } FTrainIndexerBwdDenseThdOpId;
typedef struct { uint64_t opaque; } FTrainIndexerTopKOpId;
typedef struct { uint64_t opaque; } FTrainIndexerTopKThdOpId;
typedef struct { uint64_t opaque; } FTrainCompactifyOpId;
typedef struct { uint64_t opaque; } FTrainCSACompressorFwdOpId;
typedef struct { uint64_t opaque; } FTrainCSACompressorBwdOpId;
/* mHC operators; one ID type per operator family. */
typedef struct { uint64_t opaque; } FTrainSinkhornFwdOpId;
typedef struct { uint64_t opaque; } FTrainSinkhornBwdOpId;
typedef struct { uint64_t opaque; } FTrainHAggregateFwdOpId;
typedef struct { uint64_t opaque; } FTrainHAggregateBwdOpId;
typedef struct { uint64_t opaque; } FTrainHPostBdaFwdOpId;
typedef struct { uint64_t opaque; } FTrainHPostBdaBwdOpId;
typedef struct { uint64_t opaque; } FTrainProjRmsFwdOpId;
typedef struct { uint64_t opaque; } FTrainProjRmsBwdOpId;
typedef struct { uint64_t opaque; } FTrainProjRmsComputeHFwdOpId;
typedef struct { uint64_t opaque; } FTrainProjRmsComputeHBwdOpId;

// clang-format on

/** @} */

/* Opaque resource handles. */

typedef struct FTrainPatternStruct* FTrainPattern;
typedef struct FTrainOpsStruct* FTrainOps;
typedef struct FTrainArgsStruct* FTrainArgs;
typedef struct FTrainPlanStruct* FTrainPlan;

/* Thread-local error queries. */

/**
 * @brief Returns the last API status recorded for the calling thread.
 *
 * A thread initially reports FTRAIN_STATUS_SUCCESS. Each non-query Flash Train
 * C API call replaces this status, while calls to ftrainGetLastStatus() and
 * ftrainGetLastMessage() leave it unchanged. Threads maintain independent
 * status values.
 *
 * @return The calling thread's last API status.
 */
FTRAIN_API FTrainStatus ftrainGetLastStatus(void);

/**
 * @brief Returns the last API diagnostic message recorded for the calling thread.
 *
 * The returned string is initially empty and is owned by Flash Train. The
 * caller must not modify or free it. The pointer remains valid until the same
 * thread makes another non-query Flash Train C API call or exits. Calls to
 * ftrainGetLastStatus() and ftrainGetLastMessage() do not invalidate it.
 * Threads maintain independent messages.
 *
 * @return A non-null pointer to a null-terminated diagnostic string.
 */
FTRAIN_API const char* ftrainGetLastMessage(void);

/* Pattern construction. */

/**
 * @brief Creates an empty mutable Pattern.
 *
 * @param pattern Non-null output pointer. On success, receives the created
 * Pattern. It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPatternCreate(FTrainPattern* pattern);

/**
 * @brief Destroys a Pattern.
 *
 * Destroying a Pattern does not invalidate an Ops object already created from
 * it or the operand and operation IDs captured by that Ops object.
 *
 * @param pattern Pattern to destroy.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPatternDestroy(FTrainPattern pattern);

/**
 * @brief Adds a Tensor operand to a Pattern.
 *
 * @param pattern Pattern that receives the operand.
 * @param tensor Non-null output pointer. On success, receives the created
 * Tensor role ID. It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPatternAddTensor(FTrainPattern pattern, FTrainTensorId* tensor);

/**
 * @brief Adds an ordered TensorList operand to a Pattern.
 *
 * @param pattern Pattern that receives the operand.
 * @param tensor_list Non-null output pointer. On success, receives the created
 * TensorList role ID. It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPatternAddTensorList(FTrainPattern pattern, FTrainTensorListId* tensor_list);

/**
 * @brief Adds a grouped Tensor operand to a Pattern.
 *
 * @param pattern Pattern that receives the operand.
 * @param grouped_tensor Non-null output pointer. On success, receives the
 * created GroupedTensor role ID. It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPatternAddGroupedTensor(FTrainPattern pattern, FTrainGroupedTensorId* grouped_tensor);

/**
 * @brief Adds a Gemm topology operation to a Pattern.
 *
 * @param pattern Pattern that receives the operation.
 * @param op Non-null output pointer. On success, receives the operation ID. It
 * is unchanged on failure.
 * @param a Tensor role for matrix A.
 * @param b Tensor role for matrix B.
 * @param c Tensor role for matrix C.
 * @param d Tensor role for output D.
 * @param alpha Tensor role for alpha.
 * @param beta Tensor role for beta.
 * @return Status of the operation. All role IDs must belong to pattern, and
 * the input role IDs must be distinct; pattern is unchanged on failure.
 */
FTRAIN_API FTrainStatus ftrainPatternAddGemm(FTrainPattern pattern, FTrainGemmOpId* op, FTrainTensorId a,
                                             FTrainTensorId b, FTrainTensorId c, FTrainTensorId d, FTrainTensorId alpha,
                                             FTrainTensorId beta);

/**
 * @brief Adds a GroupedABCDGemm topology operation to a Pattern.
 *
 * @param pattern Pattern that receives the operation.
 * @param op Non-null output pointer. On success, receives the operation ID. It
 * is unchanged on failure.
 * @param a GroupedTensor role for matrix A.
 * @param b GroupedTensor role for matrix B.
 * @param c GroupedTensor role for matrix C.
 * @param d GroupedTensor role for output D.
 * @param alpha Tensor role for alpha.
 * @param beta Tensor role for beta.
 * @return Status of the operation. All role IDs must belong to pattern, and
 * the input role IDs must be distinct; pattern is unchanged on failure.
 */
FTRAIN_API FTrainStatus ftrainPatternAddGroupedABCDGemm(FTrainPattern pattern, FTrainGroupedABCDGemmOpId* op,
                                                        FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                        FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                        FTrainTensorId alpha, FTrainTensorId beta);

/**
 * @brief Adds a GroupedBCDGemm topology operation to a Pattern.
 *
 * @param pattern Pattern that receives the operation.
 * @param op Non-null output pointer. On success, receives the operation ID. It
 * is unchanged on failure.
 * @param a TensorList role for matrix A.
 * @param b GroupedTensor role for matrix B.
 * @param c GroupedTensor role for matrix C.
 * @param d GroupedTensor role for output D.
 * @param alpha Tensor role for alpha.
 * @param beta Tensor role for beta.
 * @return Status of the operation. All role IDs must belong to pattern, and
 * the input role IDs must be distinct; pattern is unchanged on failure.
 */
FTRAIN_API FTrainStatus ftrainPatternAddGroupedBCDGemm(FTrainPattern pattern, FTrainGroupedBCDGemmOpId* op,
                                                       FTrainTensorListId a, FTrainGroupedTensorId b,
                                                       FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                       FTrainTensorId alpha, FTrainTensorId beta);

/**
 * @brief Adds a GroupedABGemm topology operation to a Pattern.
 *
 * @param pattern Pattern that receives the operation.
 * @param op Non-null output pointer. On success, receives the operation ID. It
 * is unchanged on failure.
 * @param a GroupedTensor role for matrix A.
 * @param b GroupedTensor role for matrix B.
 * @param c TensorList role for matrix C.
 * @param d TensorList role for output D.
 * @param alpha Tensor role for alpha.
 * @param beta Tensor role for beta.
 * @return Status of the operation. All role IDs must belong to pattern, and
 * the input role IDs must be distinct; pattern is unchanged on failure.
 */
FTRAIN_API FTrainStatus ftrainPatternAddGroupedABGemm(FTrainPattern pattern, FTrainGroupedABGemmOpId* op,
                                                      FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                      FTrainTensorListId c, FTrainTensorListId d, FTrainTensorId alpha,
                                                      FTrainTensorId beta);

/**
 * @defgroup FTrainDsaPattern DSA operator topology
 *
 * One function per DSA operator family. Every argument is a Tensor role of
 * the same pattern and, except where noted, must be distinct. Scalar roles
 * (ratio, sm_scale, eps, top_k, etc.) are rank-zero host Tensor inputs, like
 * Gemm's alpha and beta; bind them with ftrainArgsSetTensor. Integer values
 * use INT64, floating values FP32, and boolean flags UINT8. Port comments
 * name the shapes each family validates in its Problem constructor; see
 * docs/usage.md for the operator catalogue.
 * @{
 */

/** q (B,Sq,Hq,D), k (B,Sk,1,D), w (B,Sq,Hq) -> scores (B,Sq,Sk) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerFwd(FTrainPattern pattern, FTrainIndexerFwdOpId* op, FTrainTensorId q,
                                                   FTrainTensorId k, FTrainTensorId w, FTrainTensorId scores,
                                                   FTrainTensorId ratio, FTrainTensorId sm_scale);

/** THD indexer forward; q_causal_offsets' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerFwdThd(FTrainPattern pattern, FTrainIndexerFwdThdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                      FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                      FTrainTensorId q_causal_offsets, FTrainTensorId scores,
                                                      FTrainTensorId ratio, FTrainTensorId sm_scale,
                                                      FTrainTensorId max_seqlen_kv);

/** q, k, w -> out (B,Sq,Sk) f32, denom (B,Sq) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddDenseIndexerScore(FTrainPattern pattern, FTrainDenseIndexerScoreOpId* op,
                                                          FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                          FTrainTensorId out, FTrainTensorId denom,
                                                          FTrainTensorId ratio, FTrainTensorId sm_scale);

/** THD dense indexer score; q_causal_offsets' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddDenseIndexerScoreThd(FTrainPattern pattern, FTrainDenseIndexerScoreThdOpId* op,
                                                             FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                             FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                             FTrainTensorId q_causal_offsets, FTrainTensorId out,
                                                             FTrainTensorId denom, FTrainTensorId ratio,
                                                             FTrainTensorId sm_scale, FTrainTensorId max_seqlen_kv);

/** q, k, lse (B,Sq,Hq) f32 -> out (B,Sq,Sk) f32, denom (B,Sq) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddDenseAttnScore(FTrainPattern pattern, FTrainDenseAttnScoreOpId* op,
                                                       FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                       FTrainTensorId out, FTrainTensorId denom, FTrainTensorId ratio,
                                                       FTrainTensorId softmax_scale);

/** THD dense attention score; q_causal_offsets' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddDenseAttnScoreThd(FTrainPattern pattern, FTrainDenseAttnScoreThdOpId* op,
                                                          FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                          FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                          FTrainTensorId q_causal_offsets, FTrainTensorId out,
                                                          FTrainTensorId denom, FTrainTensorId ratio,
                                                          FTrainTensorId softmax_scale, FTrainTensorId max_seqlen_kv);

/** q, k, w, topk_indices (B,Sq,topk) i32 -> predict (B,Sq,topk) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddSparseIndexerPredict(FTrainPattern pattern, FTrainSparseIndexerPredictOpId* op,
                                                             FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                             FTrainTensorId topk_indices, FTrainTensorId predict,
                                                             FTrainTensorId sm_scale,
                                                             FTrainTensorId topk_indices_global);

/** q, k, lse, topk_indices -> target (B,Sq,topk) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddSparseAttnTarget(FTrainPattern pattern, FTrainSparseAttnTargetOpId* op,
                                                         FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                         FTrainTensorId topk_indices, FTrainTensorId target,
                                                         FTrainTensorId softmax_scale,
                                                         FTrainTensorId topk_indices_global);

/** Sparse attention forward; topk_length's memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddSparseAttnFwd(FTrainPattern pattern, FTrainSparseAttnFwdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId kv, FTrainTensorId attn_sink,
                                                      FTrainTensorId topk_indices, FTrainTensorId topk_length,
                                                      FTrainTensorId out, FTrainTensorId lse,
                                                      FTrainTensorId softmax_scale);

/** Sparse attention forward with the indexer region. */
FTRAIN_API FTrainStatus ftrainPatternAddSparseAttnFwdIndexer(FTrainPattern pattern, FTrainSparseAttnFwdIndexerOpId* op,
                                                             FTrainTensorId q, FTrainTensorId kv,
                                                             FTrainTensorId attn_sink, FTrainTensorId topk_indices,
                                                             FTrainTensorId topk_length, FTrainTensorId out,
                                                             FTrainTensorId max_logits, FTrainTensorId lse,
                                                             FTrainTensorId lse_indexer, FTrainTensorId softmax_scale,
                                                             FTrainTensorId indexer_topk);

/** Sparse attention backward; topk_length's memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddSparseAttnBwd(FTrainPattern pattern, FTrainSparseAttnBwdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId kv, FTrainTensorId out,
                                                      FTrainTensorId dout, FTrainTensorId lse, FTrainTensorId attn_sink,
                                                      FTrainTensorId topk_indices, FTrainTensorId topk_length,
                                                      FTrainTensorId dq, FTrainTensorId dkv, FTrainTensorId dsink,
                                                      FTrainTensorId softmax_scale);

/** Sparse indexer backward; grad_attn_score may alias attn_score.
 * BF16 grad_loss also accepts an FP32 device scalar, read on the stream. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerBwdSparse(
    FTrainPattern pattern, FTrainIndexerBwdSparseOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId index_score, FTrainTensorId topk_indices,
    FTrainTensorId grad_attn_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk, FTrainTensorId sm_scale,
    FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId topk_indices_global);

/** Dense indexer backward; grad_index_score may alias index_score. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerBwdDense(
    FTrainPattern pattern, FTrainIndexerBwdDenseOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId attn_l1norm, FTrainTensorId index_score,
    FTrainTensorId index_lse, FTrainTensorId grad_index_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk,
    FTrainTensorId sm_scale, FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId ratio);

/** THD dense indexer backward; q_causal_offsets' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerBwdDenseThd(
    FTrainPattern pattern, FTrainIndexerBwdDenseThdOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId attn_l1norm, FTrainTensorId index_score,
    FTrainTensorId index_lse, FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_k, FTrainTensorId q_causal_offsets,
    FTrainTensorId grad_index_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk, FTrainTensorId sm_scale,
    FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId ratio);

/** Per-row top-K; out_values' memory may be null when order_values is false. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerTopK(FTrainPattern pattern, FTrainIndexerTopKOpId* op,
                                                    FTrainTensorId input_values, FTrainTensorId seq_lens,
                                                    FTrainTensorId out_indices, FTrainTensorId out_values,
                                                    FTrainTensorId top_k, FTrainTensorId next_n,
                                                    FTrainTensorId order_values);

/** Per-row THD top-K; q_causal_offsets and out_values' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddIndexerTopKThd(FTrainPattern pattern, FTrainIndexerTopKThdOpId* op,
                                                       FTrainTensorId input_values, FTrainTensorId cu_seqlens_q,
                                                       FTrainTensorId cu_seqlens_kv, FTrainTensorId q_causal_offsets,
                                                       FTrainTensorId out_indices, FTrainTensorId out_values,
                                                       FTrainTensorId ratio, FTrainTensorId top_k,
                                                       FTrainTensorId order_values);

/** indices (n_rows,topk) i32 -> output i32, topk_length (n_rows) i32. */
FTRAIN_API FTrainStatus ftrainPatternAddCompactify(FTrainPattern pattern, FTrainCompactifyOpId* op,
                                                   FTrainTensorId indices, FTrainTensorId output,
                                                   FTrainTensorId topk_length);

/** CSA compressor forward. */
FTRAIN_API FTrainStatus ftrainPatternAddCSACompressorFwd(FTrainPattern pattern, FTrainCSACompressorFwdOpId* op,
                                                         FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                         FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                         FTrainTensorId compressed, FTrainTensorId ratio,
                                                         FTrainTensorId head_dim, FTrainTensorId coff,
                                                         FTrainTensorId total_comp);

/** CSA compressor backward; grad_kv/grad_score/grad_ape are zeroed by the launch. */
FTRAIN_API FTrainStatus ftrainPatternAddCSACompressorBwd(FTrainPattern pattern, FTrainCSACompressorBwdOpId* op,
                                                         FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                         FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                         FTrainTensorId grad_compressed, FTrainTensorId grad_kv,
                                                         FTrainTensorId grad_score, FTrainTensorId grad_ape,
                                                         FTrainTensorId ratio, FTrainTensorId head_dim,
                                                         FTrainTensorId coff);

/**
 * @defgroup FTrainMhcPattern mHC operator topology
 *
 * One function per mHC operator family; mHC's forward and backward are
 * separate families. Optional ports are documented per function.
 * @{
 */

/** logits (...,HC,HC) -> output, m_init (...,HC,HC) f32. */
FTRAIN_API FTrainStatus ftrainPatternAddSinkhornFwd(FTrainPattern pattern, FTrainSinkhornFwdOpId* op,
                                                    FTrainTensorId logits, FTrainTensorId output, FTrainTensorId m_init,
                                                    FTrainTensorId num_iterations, FTrainTensorId eps);

/** grad_output, m_init (f32) -> grad_logits. */
FTRAIN_API FTrainStatus ftrainPatternAddSinkhornBwd(FTrainPattern pattern, FTrainSinkhornBwdOpId* op,
                                                    FTrainTensorId grad_output, FTrainTensorId m_init,
                                                    FTrainTensorId grad_logits, FTrainTensorId num_iterations,
                                                    FTrainTensorId eps);

/** x (S,B,N,C), h_pre (S,B,N) -> aggregated (S,B,C). */
FTRAIN_API FTrainStatus ftrainPatternAddHAggregateFwd(FTrainPattern pattern, FTrainHAggregateFwdOpId* op,
                                                      FTrainTensorId x, FTrainTensorId h_pre,
                                                      FTrainTensorId aggregated);

/** grad_aggregated, x, h_pre -> grad_x, grad_h_pre. */
FTRAIN_API FTrainStatus ftrainPatternAddHAggregateBwd(FTrainPattern pattern, FTrainHAggregateBwdOpId* op,
                                                      FTrainTensorId grad_aggregated, FTrainTensorId x,
                                                      FTrainTensorId h_pre, FTrainTensorId grad_x,
                                                      FTrainTensorId grad_h_pre);

/** h_res, residual, h_post, x, bias (optional) -> output. */
FTRAIN_API FTrainStatus ftrainPatternAddHPostBdaFwd(FTrainPattern pattern, FTrainHPostBdaFwdOpId* op,
                                                    FTrainTensorId h_res, FTrainTensorId residual,
                                                    FTrainTensorId h_post, FTrainTensorId x, FTrainTensorId bias,
                                                    FTrainTensorId output);

/** h_post_bda backward; bias' and grad_bias' memory may be null. */
FTRAIN_API FTrainStatus ftrainPatternAddHPostBdaBwd(FTrainPattern pattern, FTrainHPostBdaBwdOpId* op,
                                                    FTrainTensorId grad_output, FTrainTensorId residual,
                                                    FTrainTensorId x, FTrainTensorId bias, FTrainTensorId h_res,
                                                    FTrainTensorId h_post, FTrainTensorId grad_residual,
                                                    FTrainTensorId grad_x, FTrainTensorId grad_h_res,
                                                    FTrainTensorId grad_h_post, FTrainTensorId grad_bias);

/** x (M,K), weight (N,K) -> proj (M,N), r (M,1). */
FTRAIN_API FTrainStatus ftrainPatternAddProjRmsFwd(FTrainPattern pattern, FTrainProjRmsFwdOpId* op, FTrainTensorId x,
                                                   FTrainTensorId weight, FTrainTensorId proj, FTrainTensorId r,
                                                   FTrainTensorId eps);

/** grad_proj, grad_r, x, weight, r -> grad_x, grad_weight. */
FTRAIN_API FTrainStatus ftrainPatternAddProjRmsBwd(FTrainPattern pattern, FTrainProjRmsBwdOpId* op,
                                                   FTrainTensorId grad_proj, FTrainTensorId grad_r, FTrainTensorId x,
                                                   FTrainTensorId weight, FTrainTensorId r, FTrainTensorId grad_x,
                                                   FTrainTensorId grad_weight, FTrainTensorId eps);

/** Fused projection + RMS norm + compute_h forward. */
FTRAIN_API FTrainStatus ftrainPatternAddProjRmsComputeHFwd(
    FTrainPattern pattern, FTrainProjRmsComputeHFwdOpId* op, FTrainTensorId x, FTrainTensorId weight,
    FTrainTensorId alpha_pre, FTrainTensorId alpha_post, FTrainTensorId alpha_res, FTrainTensorId bias,
    FTrainTensorId h_pre, FTrainTensorId h_post, FTrainTensorId h_res, FTrainTensorId r, FTrainTensorId proj,
    FTrainTensorId n, FTrainTensorId eps, FTrainTensorId compute_h_eps);

/** Fused projection + RMS norm + compute_h backward. */
FTRAIN_API FTrainStatus ftrainPatternAddProjRmsComputeHBwd(
    FTrainPattern pattern, FTrainProjRmsComputeHBwdOpId* op, FTrainTensorId grad_h_pre, FTrainTensorId grad_h_post,
    FTrainTensorId grad_h_res, FTrainTensorId grad_r, FTrainTensorId h_pre, FTrainTensorId h_post, FTrainTensorId proj,
    FTrainTensorId r_ch, FTrainTensorId x, FTrainTensorId weight, FTrainTensorId alpha_pre, FTrainTensorId alpha_post,
    FTrainTensorId alpha_res, FTrainTensorId grad_x, FTrainTensorId grad_weight, FTrainTensorId grad_alpha,
    FTrainTensorId grad_bias, FTrainTensorId n, FTrainTensorId eps, FTrainTensorId compute_h_eps);

/** @} */

/* Ops matching. */

/**
 * @brief Creates an immutable Ops object by matching a Pattern.
 *
 * The created Ops object retains the matched topology identity and role mapping
 * independently of the source Pattern. Only operand and operation IDs captured
 * by the Ops object may be used with Args created from it. No concrete Tensor
 * or operation parameters participate in this step.
 *
 * @param ops Non-null output pointer. On success, receives the created Ops object.
 * It is unchanged on failure.
 * @param pattern Pattern to match.
 * @return FTRAIN_STATUS_UNSUPPORTED when the supplied Pattern does not exactly
 * match any supported Pattern; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainOpsCreate(FTrainOps* ops, FTrainPattern pattern);

/**
 * @brief Destroys an Ops object.
 *
 * A Plan already created from the Ops object remains valid.
 *
 * @param ops Ops object to destroy.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainOpsDestroy(FTrainOps ops);

/* Concrete argument binding. */

/**
 * @brief Creates initially unset Args for an Ops object.
 *
 * The Args accepts only operand and operation IDs captured by the Ops object.
 * Multiple independent Args objects may be created from the same Ops object.
 * Setters may be called in any order, and setting the same role again replaces
 * its previous value. ftrainPlanCreate() checks that all required roles
 * have been set.
 *
 * @param args Non-null output pointer. On success, receives the created Args.
 * It is unchanged on failure.
 * @param ops Ops object whose roles define the parameter collection.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsCreate(FTrainArgs* args, FTrainOps ops);

/**
 * @brief Destroys Args.
 *
 * A Plan already created from the Args remains valid.
 *
 * @param args Args to destroy.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsDestroy(FTrainArgs args);

/**
 * @brief Sets or replaces the concrete storage for a Tensor role.
 *
 * The library copies the dimension and stride metadata before returning. It
 * does not copy or own storage_view.memory.
 *
 * @param args Args to update.
 * @param tensor Tensor role captured by the associated Ops object.
 * @param storage_view Concrete Tensor storage description.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetTensor(FTrainArgs args, FTrainTensorId tensor, FTrainStorageView storage_view);

/**
 * @brief Sets or replaces the concrete storages for a TensorList role.
 *
 * The library copies storage_views and their dimension and stride metadata
 * before returning. It does not copy or own the referenced memory addresses.
 *
 * @param args Args to update.
 * @param tensor_list TensorList role captured by the associated Ops object.
 * @param storage_views Ordered input array. When num_storage_views is nonzero,
 * it must contain that many readable elements. It may be null when
 * num_storage_views is zero.
 * @param num_storage_views Number of elements in storage_views.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetTensorList(FTrainArgs args, FTrainTensorListId tensor_list,
                                                const FTrainStorageView storage_views[], uint32_t num_storage_views);

/**
 * @brief Sets or replaces the concrete description for a GroupedTensor role.
 *
 * The library copies all supplied storage descriptions and their dimension and
 * stride metadata before returning. It does not copy or own their referenced
 * memory addresses. data.num_dims defines the GroupedTensor rank. Per-group
 * dimensions are described by dim_sizes rather than data.dims.
 *
 * When offsets, dim_sizes, and strides are supplied, the logical element
 * offset for index[d] in group g is:
 *
 *     offsets[g] + sum(index[d] * strides[d][g])
 *
 * where 0 <= index[d] < dim_sizes[d][g]. Metadata numeric types and host or
 * device placement are selected by the caller and checked for implementation
 * support by ftrainPlanCreate().
 *
 * @param args Args to update.
 * @param grouped_tensor GroupedTensor role captured by the associated Ops object.
 * @param num_groups Number of represented groups.
 * @param data Storage description for the grouped data.
 * @param offsets Optional storage description for a rank-one array containing
 * num_groups group offsets. Each offset is measured in logical elements from
 * data.memory. A null pointer means that no offsets description is supplied.
 * @param dim_sizes Optional dimension-major array of per-group dimension-size
 * storage descriptions. A non-null array must contain data.num_dims readable
 * elements; element d describes a rank-one array of num_groups sizes.
 * @param strides Optional dimension-major array of per-group logical-element
 * stride storage descriptions. A non-null array must contain data.num_dims
 * readable elements; element d describes a rank-one array of num_groups
 * strides.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetGroupedTensor(FTrainArgs args, FTrainGroupedTensorId grouped_tensor,
                                                   uint32_t num_groups, FTrainStorageView data,
                                                   const FTrainStorageView* offsets,
                                                   const FTrainStorageView dim_sizes[],
                                                   const FTrainStorageView strides[]);

/**
 * @brief Sets or replaces parameters for a Gemm operation role.
 *
 * @param args Args to update.
 * @param op Gemm operation role captured by the associated Ops object.
 * @param compute_type Numeric type used for computation.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetGemm(FTrainArgs args, FTrainGemmOpId op, FTrainNumericType compute_type);

/**
 * @brief Sets or replaces parameters for a GroupedABCDGemm operation role.
 *
 * @param args Args to update.
 * @param op GroupedABCDGemm operation role captured by the associated Ops object.
 * @param compute_type Numeric type used for computation.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetGroupedABCDGemm(FTrainArgs args, FTrainGroupedABCDGemmOpId op,
                                                     FTrainNumericType compute_type);

/**
 * @brief Sets or replaces parameters for a GroupedBCDGemm operation role.
 *
 * @param args Args to update.
 * @param op GroupedBCDGemm operation role captured by the associated Ops object.
 * @param compute_type Numeric type used for computation.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetGroupedBCDGemm(FTrainArgs args, FTrainGroupedBCDGemmOpId op,
                                                    FTrainNumericType compute_type);

/**
 * @brief Sets or replaces parameters for a GroupedABGemm operation role.
 *
 * @param args Args to update.
 * @param op GroupedABGemm operation role captured by the associated Ops object.
 * @param compute_type Numeric type used for computation.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainArgsSetGroupedABGemm(FTrainArgs args, FTrainGroupedABGemmOpId op,
                                                   FTrainNumericType compute_type);

/**
 * @defgroup FTrainDsaArgs DSA operator parameters
 *
 * One setter per DSA operator family. These setters carry only compute_type
 * when the family has a dtype chain; the other setters mark fixed-type
 * operations. Setters that take compute_type reject invalid values. Runtime
 * scalar Tensor values are validated by the Problem when creating a Plan.
 * @{
 */

/** compute_type is the MMAC chain (fp32, fp16, or bf16); ratio >= 1; sm_scale finite. */
FTRAIN_API FTrainStatus ftrainArgsSetIndexerFwd(FTrainArgs args, FTrainIndexerFwdOpId op,
                                                FTrainNumericType compute_type);

/** THD variant; the max_seqlen_kv Tensor gives the row width of scores. */
FTRAIN_API FTrainStatus ftrainArgsSetIndexerFwdThd(FTrainArgs args, FTrainIndexerFwdThdOpId op,
                                                   FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetDenseIndexerScore(FTrainArgs args, FTrainDenseIndexerScoreOpId op,
                                                       FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetDenseIndexerScoreThd(FTrainArgs args, FTrainDenseIndexerScoreThdOpId op,
                                                          FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetDenseAttnScore(FTrainArgs args, FTrainDenseAttnScoreOpId op,
                                                    FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetDenseAttnScoreThd(FTrainArgs args, FTrainDenseAttnScoreThdOpId op,
                                                       FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetSparseIndexerPredict(FTrainArgs args, FTrainSparseIndexerPredictOpId op,
                                                          FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetSparseAttnTarget(FTrainArgs args, FTrainSparseAttnTargetOpId op,
                                                      FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetSparseAttnFwd(FTrainArgs args, FTrainSparseAttnFwdOpId op,
                                                   FTrainNumericType compute_type);

/** The indexer_topk Tensor must satisfy 1 <= indexer_topk <= topk. */
FTRAIN_API FTrainStatus ftrainArgsSetSparseAttnFwdIndexer(FTrainArgs args, FTrainSparseAttnFwdIndexerOpId op,
                                                          FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetSparseAttnBwd(FTrainArgs args, FTrainSparseAttnBwdOpId op,
                                                   FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetIndexerBwdSparse(FTrainArgs args, FTrainIndexerBwdSparseOpId op,
                                                      FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetIndexerBwdDense(FTrainArgs args, FTrainIndexerBwdDenseOpId op,
                                                     FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetIndexerBwdDenseThd(FTrainArgs args, FTrainIndexerBwdDenseThdOpId op,
                                                        FTrainNumericType compute_type);

/** The next_n Tensor must be >= 1; order_values selects whether out_values is written. */
FTRAIN_API FTrainStatus ftrainArgsSetIndexerTopK(FTrainArgs args, FTrainIndexerTopKOpId op);

FTRAIN_API FTrainStatus ftrainArgsSetIndexerTopKThd(FTrainArgs args, FTrainIndexerTopKThdOpId op);

FTRAIN_API FTrainStatus ftrainArgsSetCompactify(FTrainArgs args, FTrainCompactifyOpId op);

/** Scalar Tensor constraints: ratio is 4 or 128; coff is 1 or 2; total_comp is the compressed row count. */
FTRAIN_API FTrainStatus ftrainArgsSetCSACompressorFwd(FTrainArgs args, FTrainCSACompressorFwdOpId op,
                                                      FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetCSACompressorBwd(FTrainArgs args, FTrainCSACompressorBwdOpId op,
                                                      FTrainNumericType compute_type);

/** @} */

/**
 * @defgroup FTrainMhcArgs mHC operator parameters
 *
 * Runtime scalars are rank-zero host Tensor ports bound with
 * ftrainArgsSetTensor. These setters carry only compute_type.
 * @{
 */

/** num_iterations >= 0; eps > 0 and finite. */
FTRAIN_API FTrainStatus ftrainArgsSetSinkhornFwd(FTrainArgs args, FTrainSinkhornFwdOpId op,
                                                 FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetSinkhornBwd(FTrainArgs args, FTrainSinkhornBwdOpId op,
                                                 FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetHAggregateFwd(FTrainArgs args, FTrainHAggregateFwdOpId op,
                                                   FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetHAggregateBwd(FTrainArgs args, FTrainHAggregateBwdOpId op,
                                                   FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetHPostBdaFwd(FTrainArgs args, FTrainHPostBdaFwdOpId op,
                                                 FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetHPostBdaBwd(FTrainArgs args, FTrainHPostBdaBwdOpId op,
                                                 FTrainNumericType compute_type);

/** eps > 0 and finite. */
FTRAIN_API FTrainStatus ftrainArgsSetProjRmsFwd(FTrainArgs args, FTrainProjRmsFwdOpId op,
                                                FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetProjRmsBwd(FTrainArgs args, FTrainProjRmsBwdOpId op,
                                                FTrainNumericType compute_type);

/** 0 < 2*n < N; eps > 0; compute_h_eps >= 0. */
FTRAIN_API FTrainStatus ftrainArgsSetProjRmsComputeHFwd(FTrainArgs args, FTrainProjRmsComputeHFwdOpId op,
                                                        FTrainNumericType compute_type);

FTRAIN_API FTrainStatus ftrainArgsSetProjRmsComputeHBwd(FTrainArgs args, FTrainProjRmsComputeHBwdOpId op,
                                                        FTrainNumericType compute_type);

/** @} */

/* Plan selection and execution. */

/**
 * @brief Creates a Plan for one complete set of concrete parameters.
 *
 * The call uses the calling thread's current device, queries the library's
 * internal execution-plan cache, performs selection when needed, and binds the
 * supplied parameters to the returned Plan. Raw memory addresses do not
 * prevent compatible execution plans from being reused internally.
 *
 * The returned Plan holds the selected Primitives ordered best-first;
 * primitive 0 is the recommended default. When a Finder contributes a
 * ranked candidate, its candidates are the selection; when none does, the
 * Plan holds every applicable registered Primitive in registration order.
 * Workspace budgets are the caller's concern: walk the Primitives with
 * ftrainPlanGetPrimitiveRequiredWorkspaceBytes() and keep the first index
 * whose requirement fits the caller's workspace; execution enforces each
 * Primitive's requirement.
 *
 * A successful call always returns a Plan holding at least one usable
 * Primitive; ftrainPlanGetNumPrimitives() is not an availability check.
 *
 * @param plan Non-null output pointer. On success, receives the created Plan.
 * It is unchanged on failure.
 * @param args Complete parameters; the engine is located through the
 * parameters' own topology.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when parameters are incomplete or
 * inconsistent; FTRAIN_STATUS_OVERFLOW when a derived element or byte count
 * cannot be represented; FTRAIN_STATUS_UNSUPPORTED when no available
 * implementation satisfies the parameters; otherwise, the status of the
 * operation.
 */
FTRAIN_API FTrainStatus ftrainPlanCreate(FTrainPlan* plan, FTrainArgs args);

/**
 * @brief Destroys a Plan.
 *
 * @param plan Plan to destroy.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPlanDestroy(FTrainPlan plan);

/**
 * @brief Returns the number of Primitives in a Plan.
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
FTRAIN_API FTrainStatus ftrainPlanGetNumPrimitives(FTrainPlan plan, uint64_t* num_primitives);

/**
 * @brief Returns the name of one Primitive in a Plan.
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
FTRAIN_API FTrainStatus ftrainPlanGetPrimitiveName(FTrainPlan plan, uint64_t primitive_index, const char** name);

/**
 * @brief Returns the workspace one Primitive of a Plan requires.
 *
 * @param plan Plan to query.
 * @param primitive_index Index of the Primitive within the Plan.
 * @param workspace_bytes Non-null output pointer that receives the required
 * size in bytes. It is unchanged on failure.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when primitive_index is out of
 * range; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPlanGetPrimitiveRequiredWorkspaceBytes(FTrainPlan plan, uint64_t primitive_index,
                                                                     uint64_t* workspace_bytes);

/**
 * @brief Executes one Primitive of a Plan on a stream.
 *
 * Every Primitive in the Plan is bound to the current device observed by
 * ftrainPlanCreate(). The supplied stream and referenced Tensor memory
 * must be compatible with that device.
 *
 * The workspace size must be at least the value returned by
 * ftrainPlanGetPrimitiveRequiredWorkspaceBytes() for the same index. A
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
FTRAIN_API FTrainStatus ftrainPlanExecutePrimitive(FTrainPlan plan, uint64_t primitive_index, void* workspace,
                                                   uint64_t workspace_bytes, FTrainStream stream);

#ifdef __cplusplus
}
#endif

#endif
