// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/primitive/hygon/gemm/fp32_gemm.hpp"

#include "flash_train/logging.hpp"
#include "flash_train/family/gemm/family.hpp"

namespace ftrain {

// The Gemm implementation catalog in registration order: that order is the
// engine's no-Finder fallback preference, so the recommended default comes
// first.
std::vector<std::shared_ptr<const Primitive<GemmProblem>>> GemmFamily::makeRecords() {
    return {std::make_shared<Fp32Gemm>()};
}

void GemmFamily::describeProblem(const GemmProblem& problem, logging::Buffer& record) {
    const auto shape = problem.getShape();
    record.format("m=%llu n=%llu k=%llu compute_type=%d a_type=%d b_type=%d c_type=%d d_type=%d a_index=%d b_index=%d",
                  static_cast<unsigned long long>(shape.m), static_cast<unsigned long long>(shape.n),
                  static_cast<unsigned long long>(shape.k), static_cast<int>(problem.getComputeType()),
                  static_cast<int>(problem.getANumericType()), static_cast<int>(problem.getBNumericType()),
                  static_cast<int>(problem.getCNumericType()), static_cast<int>(problem.getDNumericType()),
                  static_cast<int>(problem.getAIndexType()), static_cast<int>(problem.getBIndexType()));
}

}  // namespace ftrain
