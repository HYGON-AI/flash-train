// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/constraints.hpp"
#include "flash_train/plan.hpp"

namespace ftrain {

Plan::Plan(std::vector<std::unique_ptr<PrimitiveBase>>&& primitive_operands, FTrainDeviceId device_id)
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

const char* Plan::getPrimitiveName(std::uint64_t primitive_index) const {
    if (primitive_index >= primitives_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Plan primitive index %llu is out of range for %llu primitives",
                        static_cast<unsigned long long>(primitive_index),
                        static_cast<unsigned long long>(primitives_.size()));
    }
    return primitives_[primitive_index]->getName();
}

std::uint64_t Plan::getPrimitiveRequiredWorkspaceBytes(std::uint64_t primitive_index) const {
    if (primitive_index >= primitives_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Plan primitive index %llu is out of range for %llu primitives",
                        static_cast<unsigned long long>(primitive_index),
                        static_cast<unsigned long long>(primitives_.size()));
    }
    return primitives_[primitive_index]->getRequiredWorkspaceBytes();
}

void Plan::execute(std::uint64_t primitive_index, void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) {
    const FTrainDeviceId current_device_id = getCurrentDeviceId();
    if (current_device_id != device_id_) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Plan was created for device %d but the calling thread uses device %d",
                        static_cast<int>(device_id_), static_cast<int>(current_device_id));
    }
    if (primitive_index >= primitives_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Plan primitive index %llu is out of range for %llu primitives",
                        static_cast<unsigned long long>(primitive_index),
                        static_cast<unsigned long long>(primitives_.size()));
    }
    primitives_[primitive_index]->execute(Resources{workspace, workspace_bytes, stream});
}

}  // namespace ftrain
