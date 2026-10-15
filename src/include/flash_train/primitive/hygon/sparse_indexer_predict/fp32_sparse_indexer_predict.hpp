// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_FP32_SPARSE_INDEXER_PREDICT_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_FP32_SPARSE_INDEXER_PREDICT_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_sparse_indexer_predict.hip.
bool fp32SparseIndexerPredictSparseMmacSupported();
bool fp32SparseIndexerPredictSparseMmacShmFits(int query_heads, int head_dim, int selection_width);
void fp32SparseIndexerPredictZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchFp32SparseIndexerPredict(FTrainStream stream, const void* q, const void* k, const void* w,
                                    const int* topk_indices, float* predict, int* row_count, int batch,
                                    int query_length, int key_length, int query_heads, int head_dim,
                                    int selection_width, float sm_scale, bool topk_indices_global);

// One dtype chain of the SparseIndexerPredict family. The kernel reports the
// number of rows it wrote through a per-row counter array, which this
// Primitive owns as workspace: the counter region is reserved, zeroed on the
// stream, and handed to the launch.
class Fp32SparseIndexerPredict final : public Primitive<SparseIndexerPredictProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseIndexerPredictProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseIndexerPredictProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_                  = nullptr;
    const void* k_                  = nullptr;
    const void* w_                  = nullptr;
    const int* topk_indices_        = nullptr;
    float* predict_                 = nullptr;
    int batch_                      = 0;
    int query_length_               = 0;
    int key_length_                 = 0;
    int query_heads_                = 0;
    int head_dim_                   = 0;
    int selection_width_            = 0;
    float sm_scale_                 = 0.0F;
    bool topk_indices_global_       = false;
    std::uint64_t workspace_bytes_  = 0;
    std::uint64_t row_count_offset_ = 0;
};

}  // namespace ftrain

#endif
