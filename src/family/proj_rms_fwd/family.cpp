// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/proj_rms_fwd/bf16_proj_rms_fwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/fp16_proj_rms_fwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/fp32_proj_rms_fwd.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/mixed_fp32_proj_rms_fwd.hpp"
#include "flash_train/family/proj_rms_fwd/family.hpp"

namespace ftrain {

// Uniform BF16 first, then uniform FP32/FP16 and explicit mixed weight chains.
// Each record accepts exactly one activation/weight pair; both enter the Problem key.
std::vector<std::shared_ptr<const Primitive<ProjRmsFwdProblem>>> ProjRmsFwdFamily::makeRecords() {
    return {std::make_shared<Bf16ProjRmsFwd>(),
            std::make_shared<Bf16ProjRmsFwdSplit2>(),
            std::make_shared<Bf16ProjRmsFwdSplit4>(),
            std::make_shared<Bf16ProjRmsFwdSplit8>(),
            std::make_shared<Fp32ProjRmsFwd>(),
            std::make_shared<Fp32ProjRmsFwdSplit2>(),
            std::make_shared<Fp32ProjRmsFwdSplit4>(),
            std::make_shared<Fp32ProjRmsFwdSplit8>(),
            std::make_shared<Fp16ProjRmsFwd>(),
            std::make_shared<Fp16ProjRmsFwdSplit2>(),
            std::make_shared<Fp16ProjRmsFwdSplit4>(),
            std::make_shared<Fp16ProjRmsFwdSplit8>(),
            std::make_shared<MixedFp32ProjRmsFwd>(),
            std::make_shared<MixedFp32ProjRmsFwdSplit2>(),
            std::make_shared<MixedFp32ProjRmsFwdSplit4>(),
            std::make_shared<MixedFp32ProjRmsFwdSplit8>()};
}

}  // namespace ftrain
