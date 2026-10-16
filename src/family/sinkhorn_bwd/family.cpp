// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/fp16_sinkhorn_bwd.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/fp32_sinkhorn_bwd.hpp"
#include "flash_train/family/sinkhorn_bwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SinkhornBwdProblem>>> SinkhornBwdFamily::makeRecords() {
    return {std::make_shared<Bf16SinkhornBwd>(), std::make_shared<Fp32SinkhornBwd>(),
            std::make_shared<Fp16SinkhornBwd>()};
}

}  // namespace ftrain
