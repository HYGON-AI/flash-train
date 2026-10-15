// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/bf16_sparse_attn_fwd_indexer.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/fp16_sparse_attn_fwd_indexer.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/fp32_sparse_attn_fwd_indexer.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/mixed_fp32_sparse_attn_fwd_indexer.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseAttnFwdIndexerProblem>>> SparseAttnFwdIndexerFamily::makeRecords() {
    return {std::make_shared<Bf16SparseAttnFwdIndexer>(), std::make_shared<Fp32SparseAttnFwdIndexer>(),
            std::make_shared<Fp16SparseAttnFwdIndexer>(), std::make_shared<MixedFp32SparseAttnFwdIndexer>()};
}

}  // namespace ftrain
