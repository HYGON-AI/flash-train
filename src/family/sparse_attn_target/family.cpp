// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/fp16_sparse_attn_target.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/fp32_sparse_attn_target.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/mixed_fp32_sparse_attn_target.hpp"
#include "flash_train/family/sparse_attn_target/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseAttnTargetProblem>>> SparseAttnTargetFamily::makeRecords() {
    return {std::make_shared<Bf16SparseAttnTarget>(), std::make_shared<Fp32SparseAttnTarget>(),
            std::make_shared<Fp16SparseAttnTarget>(), std::make_shared<MixedFp32SparseAttnTarget>()};
}

}  // namespace ftrain
