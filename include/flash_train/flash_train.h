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


typedef struct FTrainCollectiveContextStruct* FTrainCollectiveContext;
typedef struct FTrainAGGroupGemmStruct* FTrainAGGroupGemm;
typedef struct FTrainMoEReduceRSStruct* FTrainMoEReduceRS;
typedef struct { uint64_t opaque; } FTrainAGGroupGemmOpId;
typedef struct { uint64_t opaque; } FTrainMoEReduceRSOpId;

typedef FTrainStatus (*FTrainByteAllGather)(void* user_data, const void* send, void* receive, uint64_t bytes);

typedef struct {
    FTrainStorageView storage;
    const void* identity;
    uint64_t version;
} FTrainMoeTensor;

typedef struct {
    int32_t m_dim;
    int32_t capacity_m;
    int32_t n_dim;
    int32_t k_dim;
    int32_t num_experts;
    int32_t topk;
    const int64_t* rank_m_sizes;
    uint32_t num_rank_m_sizes;
    const char* route;
} FTrainAGGroupGemmConfig;

typedef struct {
    int32_t m_dim;
    int32_t capacity_m;
    int32_t n_dim;
    int32_t k_dim;
    int32_t num_experts;
    int32_t topk;
    const int64_t* rank_m_sizes;
    uint32_t num_rank_m_sizes;
    const char* route;
    int32_t n_chunks;
} FTrainMoEReduceRSConfig;

FTRAIN_API FTrainStatus ftrainMoeIsAvailable(bool* available);

FTRAIN_API FTrainStatus ftrainCollectiveContextCreate(FTrainCollectiveContext* context, int32_t rank,
                                                      int32_t world_size, FTrainDeviceId device,
                                                      FTrainByteAllGather allgather, void* user_data);
FTRAIN_API FTrainStatus ftrainCollectiveContextDestroy(FTrainCollectiveContext context);

FTRAIN_API FTrainStatus ftrainAGGroupGemmConfigInit(FTrainAGGroupGemmConfig* config);
FTRAIN_API FTrainStatus ftrainAGGroupGemmCreate(FTrainAGGroupGemm* operation, FTrainCollectiveContext context,
                                               const FTrainAGGroupGemmConfig* config, FTrainStream stream);
FTRAIN_API FTrainStatus ftrainAGGroupGemmExecute(FTrainAGGroupGemm operation, FTrainMoeTensor input,
                                                FTrainMoeTensor weight, FTrainMoeTensor routing,
                                                FTrainMoeTensor output, FTrainStream stream);
FTRAIN_API FTrainStatus ftrainAGGroupGemmSetRankMSizes(FTrainAGGroupGemm operation, int32_t m_dim,
                                                      const int64_t* rank_m_sizes, uint32_t num_rank_m_sizes);
FTRAIN_API FTrainStatus ftrainAGGroupGemmSynchronize(FTrainAGGroupGemm operation);
FTRAIN_API FTrainStatus ftrainAGGroupGemmDestroy(FTrainAGGroupGemm operation);

FTRAIN_API FTrainStatus ftrainMoEReduceRSConfigInit(FTrainMoEReduceRSConfig* config);
FTRAIN_API FTrainStatus ftrainMoEReduceRSCreate(FTrainMoEReduceRS* operation, FTrainCollectiveContext context,
                                               const FTrainMoEReduceRSConfig* config, FTrainStream stream);
FTRAIN_API FTrainStatus ftrainMoEReduceRSExecute(FTrainMoEReduceRS operation, FTrainMoeTensor input,
                                                FTrainMoeTensor weight, FTrainMoeTensor routing,
                                                FTrainMoeTensor expert_weight, FTrainMoeTensor output,
                                                FTrainStream stream);
FTRAIN_API FTrainStatus ftrainMoEReduceRSSetRankMSizes(FTrainMoEReduceRS operation, int32_t m_dim,
                                                      const int64_t* rank_m_sizes, uint32_t num_rank_m_sizes);
FTRAIN_API FTrainStatus ftrainMoEReduceRSSynchronize(FTrainMoEReduceRS operation);
FTRAIN_API FTrainStatus ftrainMoEReduceRSDestroy(FTrainMoEReduceRS operation);

FTRAIN_API FTrainStatus ftrainPatternAddAGGroupGemm(FTrainPattern pattern, FTrainAGGroupGemmOpId* op,
                                                   FTrainTensorId input, FTrainTensorId weight,
                                                   FTrainTensorId routing, FTrainTensorId output);
FTRAIN_API FTrainStatus ftrainArgsSetAGGroupGemm(FTrainArgs args, FTrainAGGroupGemmOpId op,
                                                FTrainAGGroupGemm operation, uint64_t weight_version,
                                                uint64_t routing_version);
FTRAIN_API FTrainStatus ftrainPatternAddMoEReduceRS(FTrainPattern pattern, FTrainMoEReduceRSOpId* op,
                                                   FTrainTensorId input, FTrainTensorId weight,
                                                   FTrainTensorId routing, FTrainTensorId expert_weight,
                                                   FTrainTensorId output);
FTRAIN_API FTrainStatus ftrainArgsSetMoEReduceRS(FTrainArgs args, FTrainMoEReduceRSOpId op,
                                                FTrainMoEReduceRS operation, uint64_t weight_version,
                                                uint64_t routing_version);

#ifdef __cplusplus
}
#endif

#endif
