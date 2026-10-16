// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/compactify/int32_compactify.hpp"
#include "flash_train/family/compactify/family.hpp"

namespace ftrain {

// A single implementation: fixed INT32 in and out.
std::vector<std::shared_ptr<const Primitive<CompactifyProblem>>> CompactifyFamily::makeRecords() {
    return {std::make_shared<Int32Compactify>()};
}

}  // namespace ftrain
