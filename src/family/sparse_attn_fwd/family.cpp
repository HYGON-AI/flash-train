// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_attn_fwd/bf16_sparse_attn_fwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/fp16_sparse_attn_fwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/fp32_sparse_attn_fwd.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/mixed_fp32_sparse_attn_fwd.hpp"
#include "flash_train/family/sparse_attn_fwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseAttnFwdProblem>>> SparseAttnFwdFamily::makeRecords() {
    return {std::make_shared<Bf16SparseAttnFwd>(), std::make_shared<Fp32SparseAttnFwd>(),
            std::make_shared<Fp16SparseAttnFwd>(), std::make_shared<MixedFp32SparseAttnFwd>()};
}

}  // namespace ftrain
