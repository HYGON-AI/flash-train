// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sinkhorn_fwd/bf16_sinkhorn_fwd.hpp"
#include "flash_train/primitive/hygon/sinkhorn_fwd/fp16_sinkhorn_fwd.hpp"
#include "flash_train/primitive/hygon/sinkhorn_fwd/fp32_sinkhorn_fwd.hpp"
#include "flash_train/family/sinkhorn_fwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SinkhornFwdProblem>>> SinkhornFwdFamily::makeRecords() {
    return {std::make_shared<Bf16SinkhornFwd>(), std::make_shared<Fp32SinkhornFwd>(),
            std::make_shared<Fp16SinkhornFwd>()};
}

}  // namespace ftrain
