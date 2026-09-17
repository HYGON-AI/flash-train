#include "flash_train/engine/base.hpp"

#include <utility>

#include "flash_train/error.hpp"

namespace ftrain {

OpsEngineBase::OpsEngineBase(const Pattern& supported_pattern) : supported_pattern_(supported_pattern) {}

OpsEngineBase::~OpsEngineBase() = default;

Plan OpsEngineBase::createPlan(const Args& args, std::uint64_t max_workspace_bytes) const {
    const FTrainDeviceId device_id = getCurrentDeviceId();
    const SelectionContext context(device_id, max_workspace_bytes);
    return Plan(device_id, createPrimitives(args, context));
}

}  // namespace ftrain
