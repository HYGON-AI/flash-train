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

#ifdef __cplusplus
}
#endif

#endif
