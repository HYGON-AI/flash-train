// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "flash_train/session/distributed.hpp"

#include "flash_train/error.hpp"

namespace ftrain {

DistributedSessionRegistry& DistributedSessionRegistry::get() {
    static DistributedSessionRegistry registry;
    return registry;
}

void DistributedSessionRegistry::registerAGGroupGemm(std::uint64_t session_id,
                                                     std::shared_ptr<AGGroupGemmSession> session) {
    if (session_id == 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm session id is zero"); }
    if (session == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm session is null"); }

    std::lock_guard<std::mutex> lock(mutex_);
    ag_sessions_.insert_or_assign(session_id, std::move(session));
}

void DistributedSessionRegistry::registerMoEReduceRS(std::uint64_t session_id,
                                                     std::shared_ptr<MoEReduceRSSession> session) {
    if (session_id == 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS session id is zero"); }
    if (session == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS session is null"); }

    std::lock_guard<std::mutex> lock(mutex_);
    moe_sessions_.insert_or_assign(session_id, std::move(session));
}

std::shared_ptr<AGGroupGemmSession> DistributedSessionRegistry::getAGGroupGemm(std::uint64_t session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = ag_sessions_.find(session_id);
    return iter == ag_sessions_.end() ? nullptr : iter->second;
}

std::shared_ptr<MoEReduceRSSession> DistributedSessionRegistry::getMoEReduceRS(std::uint64_t session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = moe_sessions_.find(session_id);
    return iter == moe_sessions_.end() ? nullptr : iter->second;
}

void DistributedSessionRegistry::eraseAGGroupGemm(std::uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    static_cast<void>(ag_sessions_.erase(session_id));
}

void DistributedSessionRegistry::eraseMoEReduceRS(std::uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    static_cast<void>(moe_sessions_.erase(session_id));
}

}  // namespace ftrain
