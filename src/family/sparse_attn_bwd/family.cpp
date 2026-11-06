// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/fp16_sparse_attn_bwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/fp32_sparse_attn_bwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/mixed_fp32_sparse_attn_bwd.hpp"
#include "flash_train/family/sparse_attn_bwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseAttnBwdProblem>>> SparseAttnBwdFamily::makeRecords() {
    return {std::make_shared<Bf16SparseAttnBwd>(),
            std::make_shared<Bf16SparseAttnBwdPacked>(),
            std::make_shared<Bf16SparseAttnBwdCached>(),
            std::make_shared<Bf16SparseAttnBwdCachedLowMemory>(),
            std::make_shared<Bf16SparseAttnBwdRegisterFused>(),
            std::make_shared<Bf16SparseAttnBwdPackedSplit>(),
            std::make_shared<Bf16SparseAttnBwdCachedNoPrefetch>(),
            std::make_shared<Bf16SparseAttnBwdCachedLocal>(),
            std::make_shared<Bf16SparseAttnBwdCachedLocalAligned>(),
            std::make_shared<Bf16SparseAttnBwdCachedGrouped>(),
            std::make_shared<Bf16SparseAttnBwdCachedCompact>(),
            std::make_shared<Bf16SparseAttnBwdCachedGlmPair>(),
            std::make_shared<Bf16SparseAttnBwdCachedGlmLocal>(),
            std::make_shared<Bf16SparseAttnBwdCachedPadded>(),
            std::make_shared<Bf16SparseAttnBwdCachedGlobalIndices>(),
            std::make_shared<Bf16SparseAttnBwdCachedNoPrefetchGlobalIndices>(),
            std::make_shared<Bf16SparseAttnBwdCachedGlm>(),
            std::make_shared<Bf16SparseAttnBwdCachedGlmGlobalIndices>(),
            std::make_shared<Fp32SparseAttnBwd>(),
            std::make_shared<Fp16SparseAttnBwd>(),
            std::make_shared<MixedFp32SparseAttnBwd>()};
}

}  // namespace ftrain
