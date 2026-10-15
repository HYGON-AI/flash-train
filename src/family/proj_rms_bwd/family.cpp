// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/proj_rms_bwd/bf16_proj_rms_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/fp16_proj_rms_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/fp32_proj_rms_bwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/mixed_fp32_proj_rms_bwd.hpp"
#include "flash_train/family/proj_rms_bwd/family.hpp"

namespace ftrain {

// Uniform BF16 first, then uniform FP32/FP16 and explicit mixed weight chains.
// Each record accepts exactly one activation/weight pair; both enter the Problem key.
std::vector<std::shared_ptr<const Primitive<ProjRmsBwdProblem>>> ProjRmsBwdFamily::makeRecords() {
    return {std::make_shared<Bf16ProjRmsBwd>(), std::make_shared<Fp32ProjRmsBwd>(), std::make_shared<Fp16ProjRmsBwd>(),
            std::make_shared<MixedFp32ProjRmsBwd>()};
}

}  // namespace ftrain
