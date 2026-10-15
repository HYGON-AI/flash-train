// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_fwd_thd/bf16_indexer_fwd_thd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd_thd/fp16_indexer_fwd_thd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd_thd/fp32_indexer_fwd_thd.hpp"
#include "flash_train/primitive/hygon/indexer_fwd_thd/mixed_fp32_indexer_fwd_thd.hpp"
#include "flash_train/family/indexer_fwd_thd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<IndexerFwdThdProblem>>> IndexerFwdThdFamily::makeRecords() {
    return {std::make_shared<Bf16IndexerFwdThd>(), std::make_shared<Fp32IndexerFwdThd>(),
            std::make_shared<Fp16IndexerFwdThd>(), std::make_shared<MixedFp32IndexerFwdThd>()};
}

}  // namespace ftrain
