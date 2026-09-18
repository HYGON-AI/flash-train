// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "flash_train/primitive/gemm/fp32_gemm.hpp"

#include "flash_train/family/gemm/family.hpp"

namespace ftrain {

// The Gemm implementation catalog in registration order: that order is the
// engine's no-Finder fallback preference, so the recommended default comes
// first.
std::vector<std::shared_ptr<const Primitive<GemmProblem>>> GemmFamily::makeRecords() {
    return {std::make_shared<Fp32Gemm>()};
}

}  // namespace ftrain
