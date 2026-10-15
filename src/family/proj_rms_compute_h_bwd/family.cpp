// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/bf16_proj_rms_compute_h_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/fp16_proj_rms_compute_h_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/fp32_proj_rms_compute_h_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/mixed_fp32_proj_rms_compute_h_bwd.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/family.hpp"

namespace ftrain {

// Uniform BF16 first, then uniform FP32/FP16 and explicit mixed weight chains.
// Each record accepts exactly one activation/weight pair; both enter the Problem key.
std::vector<std::shared_ptr<const Primitive<ProjRmsComputeHBwdProblem>>> ProjRmsComputeHBwdFamily::makeRecords() {
    return {std::make_shared<Bf16ProjRmsComputeHBwd>(), std::make_shared<Fp32ProjRmsComputeHBwd>(),
            std::make_shared<Fp16ProjRmsComputeHBwd>(), std::make_shared<MixedFp32ProjRmsComputeHBwd>()};
}

}  // namespace ftrain
