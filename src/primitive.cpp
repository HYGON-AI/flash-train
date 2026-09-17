// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "flash_train/primitive.hpp"

#include "flash_train/error.hpp"

namespace ftrain {

PrimitiveBase::~PrimitiveBase() = default;

void PrimitiveBase::execute(const Resources& resources) {
    const std::uint64_t required_workspace_bytes = getRequiredWorkspaceBytes();
    if (resources.getWorkspaceBytes() < required_workspace_bytes) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Primitive requires %llu workspace bytes but received %llu",
                        static_cast<unsigned long long>(required_workspace_bytes),
                        static_cast<unsigned long long>(resources.getWorkspaceBytes()));
    }
    if (required_workspace_bytes != 0 && resources.getWorkspace() == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Primitive requires a non-null workspace for %llu bytes",
                        static_cast<unsigned long long>(required_workspace_bytes));
    }

    executeImpl(resources);
}

}  // namespace ftrain
