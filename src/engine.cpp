// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>

#include "flash_train/family/gemm/family.hpp"
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
// and register by adding one factory here.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines() { return {makeGemmOpsEngine()}; }

}  // namespace ftrain
