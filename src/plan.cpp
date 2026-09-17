#include "flash_train/plan.hpp"

#include <algorithm>
#include <utility>

#include "flash_train/engine/base.hpp"
#include "flash_train/error.hpp"

namespace ftrain {

Plan::Plan(FTrainDeviceId device_id, std::vector<std::unique_ptr<PrimitiveBase>>&& primitive_operands)
    : device_id_(device_id), primitives_(std::move(primitive_operands)) {
    if (primitives_.empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Plan requires at least one Primitive");
    }
    for (const std::unique_ptr<PrimitiveBase>& primitive : primitives_) {
        if (primitive == nullptr) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Plan Primitive must not be null");
        }
    }
}

std::uint64_t Plan::getRequiredWorkspaceBytes() const noexcept {
    std::uint64_t required_workspace_bytes = 0;
    for (const std::unique_ptr<PrimitiveBase>& primitive : primitives_) {
        required_workspace_bytes = std::max(required_workspace_bytes, primitive->getRequiredWorkspaceBytes());
    }
    return required_workspace_bytes;
}

void Plan::execute(void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) {
    const FTrainDeviceId current_device_id = getCurrentDeviceId();
    if (current_device_id != device_id_) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Plan was created for device %d but the calling thread uses device %d",
                        static_cast<int>(device_id_), static_cast<int>(current_device_id));
    }
    for (const std::unique_ptr<PrimitiveBase>& primitive : primitives_) {
        primitive->execute(workspace, workspace_bytes, stream);
    }
}

}  // namespace ftrain
