// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/h_post_bda_fwd/bf16_h_post_bda_fwd.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp32_h_post_bda_fwd.hpp"
#include "flash_train/family/h_post_bda_fwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<HPostBdaFwdProblem>>> HPostBdaFwdFamily::makeRecords() {
    return {std::make_shared<Bf16HPostBdaFwd>(), std::make_shared<Fp32HPostBdaFwd>(),
            std::make_shared<Fp16HPostBdaFwd>()};
}

}  // namespace ftrain
