// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_fwd/bf16_indexer_fwd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd/fp16_indexer_fwd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd/fp32_indexer_fwd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd/mixed_fp32_indexer_fwd.hpp"
#include "flash_train/family/indexer_fwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<IndexerFwdProblem>>> IndexerFwdFamily::makeRecords() {
    return {std::make_shared<Bf16IndexerFwdMultiQuery>(), std::make_shared<Bf16IndexerFwdTransposed>(),
            std::make_shared<Bf16IndexerFwd>(),           std::make_shared<Bf16IndexerFwdLargeD>(),
            std::make_shared<Bf16IndexerFwdHoisted>(),    std::make_shared<Bf16IndexerFwdPrefetch>(),
            std::make_shared<Bf16IndexerFwdScoreRing>(),  std::make_shared<Fp32IndexerFwd>(),
            std::make_shared<Fp16IndexerFwd>(),           std::make_shared<MixedFp32IndexerFwd>()};
}

}  // namespace ftrain
