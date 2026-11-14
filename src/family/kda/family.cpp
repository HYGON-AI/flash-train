// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/primitive/hygon/kda/g2_kda.hpp"

#include "flash_train/family/kda/family.hpp"

namespace ftrain {

// Registration order is the no-Finder preference: the G2 gfx936 implementation first.
std::vector<std::shared_ptr<const Primitive<KdaForwardProblem>>> KdaForwardFamily::makeRecords() {
    return {std::make_shared<G2KdaForward>()};
}

std::vector<std::shared_ptr<const Primitive<KdaBackwardProblem>>> KdaBackwardFamily::makeRecords() {
    return {std::make_shared<G2KdaBackward>()};
}

}  // namespace ftrain
