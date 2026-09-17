#include "flash_train/engine/base.hpp"

namespace ftrain {

OpsEngineBase::OpsEngineBase(const Pattern& supported_pattern) : supported_pattern_(supported_pattern) {}

OpsEngineBase::~OpsEngineBase() = default;

Plan OpsEngineBase::createPlan(const Args& args, std::uint64_t max_workspace_bytes) const {
    const FTrainDeviceId device_id = getCurrentDeviceId();
    const Constraints constraints(device_id, max_workspace_bytes);
    return Plan(createPrimitives(args, constraints), device_id);
}

}  // namespace ftrain
