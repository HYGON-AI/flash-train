// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/fp16_indexer_bwd_dense_thd.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/fp32_indexer_bwd_dense_thd.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/mixed_fp32_indexer_bwd_dense_thd.hpp"
#include "flash_train/family/indexer_bwd_dense_thd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<IndexerBwdDenseThdProblem>>> IndexerBwdDenseThdFamily::makeRecords() {
    return {std::make_shared<Bf16IndexerBwdDenseThd>(), std::make_shared<Fp32IndexerBwdDenseThd>(),
            std::make_shared<Fp16IndexerBwdDenseThd>(), std::make_shared<MixedFp32IndexerBwdDenseThd>()};
}

}  // namespace ftrain
