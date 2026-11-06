// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/family/gemm/finder.hpp"
#include "flash_train/family/gemm/family.hpp"
#include "flash_train/family/ag_group_gemm/finder.hpp"
#include "flash_train/family/ag_group_gemm/family.hpp"
#include "flash_train/family/moe_reduce_rs/finder.hpp"
#include "flash_train/family/moe_reduce_rs/family.hpp"
#include "flash_train/engine.hpp"

namespace ftrain {

OpsEngineBase::OpsEngineBase(const Pattern& supported_pattern) : supported_pattern_(supported_pattern) {}

OpsEngineBase::~OpsEngineBase() = default;

Plan OpsEngineBase::createPlan(const Args& args) const {
    const FTrainDeviceId device_id = getCurrentDeviceId();
    return Plan(createPrimitives(args, Constraints{device_id}), device_id);
}

// The built-in engine catalog and the single place that enumerates the
// concrete operator families. Families add themselves under family/<op>/
// and register by constructing their engine here.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines() {
    return {std::make_shared<OpsEngine<GemmFamily, RegistrationOrderFinder>>(),
            std::make_shared<OpsEngine<AGGroupGemmFamily, AGGroupGemmRouteFinder>>(),
            std::make_shared<OpsEngine<MoEReduceRSFamily, MoEReduceRSRouteFinder>>()};
}

}  // namespace ftrain
