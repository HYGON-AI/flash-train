// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/dense_indexer_score/bf16_dense_indexer_score.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score/fp16_dense_indexer_score.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score/fp32_dense_indexer_score.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score/mixed_fp32_dense_indexer_score.hpp"
#include "flash_train/family/dense_indexer_score/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<DenseIndexerScoreProblem>>> DenseIndexerScoreFamily::makeRecords() {
    return {std::make_shared<Bf16DenseIndexerScore>(), std::make_shared<Fp32DenseIndexerScore>(),
            std::make_shared<Fp16DenseIndexerScore>(), std::make_shared<MixedFp32DenseIndexerScore>()};
}

}  // namespace ftrain
