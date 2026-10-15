// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score_thd/fp16_dense_indexer_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score_thd/fp32_dense_indexer_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score_thd/mixed_fp32_dense_indexer_score_thd.hpp"
#include "flash_train/family/dense_indexer_score_thd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<DenseIndexerScoreThdProblem>>> DenseIndexerScoreThdFamily::makeRecords() {
    return {std::make_shared<Bf16DenseIndexerScoreThd>(), std::make_shared<Fp32DenseIndexerScoreThd>(),
            std::make_shared<Fp16DenseIndexerScoreThd>(), std::make_shared<MixedFp32DenseIndexerScoreThd>()};
}

}  // namespace ftrain
