// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/dense_attn_score_thd/bf16_dense_attn_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_attn_score_thd/fp16_dense_attn_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_attn_score_thd/fp32_dense_attn_score_thd.hpp"
#include "flash_train/primitive/hygon/dense_attn_score_thd/mixed_fp32_dense_attn_score_thd.hpp"
#include "flash_train/family/dense_attn_score_thd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<DenseAttnScoreThdProblem>>> DenseAttnScoreThdFamily::makeRecords() {
    return {std::make_shared<Bf16DenseAttnScoreThd>(), std::make_shared<Fp32DenseAttnScoreThd>(),
            std::make_shared<Fp16DenseAttnScoreThd>(), std::make_shared<MixedFp32DenseAttnScoreThd>()};
}

}  // namespace ftrain
