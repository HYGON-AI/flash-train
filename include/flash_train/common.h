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

/* Plan selection and execution. */

/**
 * @brief Creates a Plan for one complete set of concrete parameters.
 *
 * The call uses the calling thread's current device, queries the library's
 * internal execution-plan cache, performs selection when needed, and binds the
 * supplied parameters to the returned Plan. Raw memory addresses do not
 * prevent compatible execution plans from being reused internally.
 *
 * Primitive 0 of the returned Plan is the recommended default: the
 * performance-optimal Primitive produced by selection. When neither the
 * cache nor any Finder yields a Primitive -- for example while
 * FTRAIN_DISABLE_SELECTION_CACHE and FTRAIN_DISABLED_FINDERS are set for
 * Finder development -- the Plan instead holds every applicable registered
 * Primitive in registration order.
 *
 * @param plan Non-null output pointer. On success, receives the created Plan.
 * It is unchanged on failure.
 * @param args Complete parameters; the engine is located through the
 * parameters' own topology.
 * @param max_ws_bytes Maximum workspace permitted during selection.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when parameters are incomplete or
 * inconsistent; FTRAIN_STATUS_OVERFLOW when a derived element or byte count
 * cannot be represented; FTRAIN_STATUS_UNSUPPORTED when no available
 * implementation satisfies the parameters and workspace limit; otherwise,
 * the status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPlanCreate(FTrainPlan* plan, FTrainArgs args, uint64_t max_ws_bytes);

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
 * @param plan Plan to query.
 * @param num_primitives Non-null output pointer that receives the count.
 * It is unchanged on failure.
 * @return Status of the operation.
 */
FTRAIN_API FTrainStatus ftrainPlanGetNumPrimitives(FTrainPlan plan, uint64_t* num_primitives);

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
FTRAIN_API FTrainStatus ftrainPlanExecute(FTrainPlan plan, uint64_t primitive_index, void* workspace,
                                          uint64_t workspace_bytes, FTrainStream stream);

#ifdef __cplusplus
}
#endif

#endif
