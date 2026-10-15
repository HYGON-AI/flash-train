// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_indexer_predict/bf16_sparse_indexer_predict.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/fp16_sparse_indexer_predict.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/fp32_sparse_indexer_predict.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/mixed_fp32_sparse_indexer_predict.hpp"
#include "flash_train/family/sparse_indexer_predict/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseIndexerPredictProblem>>> SparseIndexerPredictFamily::makeRecords() {
    return {std::make_shared<Bf16SparseIndexerPredict>(), std::make_shared<Fp32SparseIndexerPredict>(),
            std::make_shared<Fp16SparseIndexerPredict>(), std::make_shared<MixedFp32SparseIndexerPredict>()};
}

}  // namespace ftrain
