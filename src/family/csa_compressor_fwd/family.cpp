// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.hpp"
#include "flash_train/family/csa_compressor_fwd/family.hpp"

namespace ftrain {

// A single implementation: the kernel's element types are fixed.
std::vector<std::shared_ptr<const Primitive<CSACompressorFwdProblem>>> CSACompressorFwdFamily::makeRecords() {
    return {std::make_shared<Bf16CSACompressorFwd>()};
}

}  // namespace ftrain
