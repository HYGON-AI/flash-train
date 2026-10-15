// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/h_post_bda_bwd/bf16_h_post_bda_bwd.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp16_h_post_bda_bwd.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.hpp"
#include "flash_train/family/h_post_bda_bwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<HPostBdaBwdProblem>>> HPostBdaBwdFamily::makeRecords() {
    return {std::make_shared<Bf16HPostBdaBwd>(), std::make_shared<Fp32HPostBdaBwd>(),
            std::make_shared<Fp16HPostBdaBwd>()};
}

}  // namespace ftrain
