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
 * memory address. When passed to an operator entry point, Flash Train copies
 * the dimension and stride metadata before returning but does not copy or own
 * the referenced memory.
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

/* Opaque resource handles. */

typedef struct FTrainSparseAttnBwdPlanStruct* FTrainSparseAttnBwdPlan;

/* Operator arguments. */

/**
 * @brief Describes the Tensors and attributes of one sparse-attention
 * backward call.
 *
 * Let T=query tokens, S=KV tokens, H=query heads, D=Q/K width, V=value
 * width, and K=selection width.
 */
typedef struct {
    /** Device query Tensor: BF16 (T,H,D). */
    FTrainStorageView q;
    /** Device shared KV Tensor: BF16 (S,D) or (S,1,D). */
    FTrainStorageView kv;
    /** Device sink-inclusive forward output: BF16 (T,H,V). */
    FTrainStorageView out;
    /** Device incoming gradient of out: BF16 (T,H,V). */
    FTrainStorageView dout;
    /** Device KV-only logsumexp: FP32 (T,H), excluding the sink logit. */
    FTrainStorageView lse;
    /** Device sink logits: FP32 (H). */
    FTrainStorageView attn_sink;
    /**
     * Device selected KV row indices: INT32 (T,K), global row indices in the
     * shared KV tensor. Indices outside [0,S) are masked; duplicates
     * accumulate gradients.
     */
    FTrainStorageView topk_indices;
    /**
     * Device per-token valid entry counts: INT32 (T). Null memory consumes
     * all K entries; otherwise each row consumes clamp(topk_length[t],0,K)
     * entries. Even when memory is null, this view must describe an INT32
     * device tensor of shape (T).
     */
    FTrainStorageView topk_length;
    /** Device output gradient of q: BF16 (T,H,D), shaped like q. */
    FTrainStorageView dq;
    /** Device output gradient of kv: BF16, shaped like kv. */
    FTrainStorageView dkv;
    /** Device output reduced gradient of attn_sink: FP32 (H). */
    FTrainStorageView dsink;
    /** Multiplier applied to each q/kv dot product before the softmax. */
    float softmax_scale;
    /** Numeric type used for computation. */
    FTrainNumericType compute_type;
} FTrainSparseAttnBwdArgs;

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

/* Plan selection and execution. */

/**
 * @brief Selects implementations for sparse-attention backward and returns
 * them as a reusable Plan bound to the storage described by args.
 *
 * Every Tensor in args uses FTRAIN_INDEX_TYPE_CONTINUOUS with null strides.
 * Required data memory must be non-null; q/kv/out/dout/dq/dkv require 16-byte
 * alignment, and the other Tensors require their element-type alignment.
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
 * the supplied storage. Selections are keyed by Tensor metadata, never by
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
 * check. The library copies args and every dims array it points to; the
 * caller may release them on return. The described device memory is neither
 * copied nor owned and must stay valid until every execution on the returned
 * Plan completes.
 *
 * @param plan Non-null output pointer. On success, receives the created Plan.
 * It is unchanged on failure.
 * @param args Non-null pointer to the call's arguments.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when plan or args is null,
 * compute_type is INVALID, COUNT, or out of range, a Tensor view has invalid
 * enums/dimensions/metadata, or a Tensor has the wrong rank/shape/placement;
 * FTRAIN_STATUS_OVERFLOW when a tensor's byte extent cannot be represented in
 * uint64_t; FTRAIN_STATUS_UNSUPPORTED when no available implementation
 * satisfies the arguments; otherwise, the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwdCreate(FTrainSparseAttnBwdPlan* plan, const FTrainSparseAttnBwdArgs* args);

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
 * asynchronous. Every referenced Tensor memory region and workspace must
 * remain valid until that work completes; the Plan may be destroyed once
 * this function returns. The caller must externally synchronize concurrent
 * executions of the same Plan.
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

#ifdef __cplusplus
}
#endif

#endif
