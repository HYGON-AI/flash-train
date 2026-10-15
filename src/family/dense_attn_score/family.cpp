// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/dense_attn_score/bf16_dense_attn_score.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/fp16_dense_attn_score.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/fp32_dense_attn_score.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/mixed_fp32_dense_attn_score.hpp"
#include "flash_train/family/dense_attn_score/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<DenseAttnScoreProblem>>> DenseAttnScoreFamily::makeRecords() {
    return {std::make_shared<Bf16DenseAttnScore>(), std::make_shared<Fp32DenseAttnScore>(),
            std::make_shared<Fp16DenseAttnScore>(), std::make_shared<MixedFp32DenseAttnScore>()};
}

}  // namespace ftrain
