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
    return {std::make_shared<Bf16SparseAttnBwd>(), std::make_shared<Fp32SparseAttnBwd>(),
            std::make_shared<Fp16SparseAttnBwd>(), std::make_shared<MixedFp32SparseAttnBwd>()};
}

}  // namespace ftrain
