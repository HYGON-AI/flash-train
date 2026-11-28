// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_CONSTRAINTS_HPP_
#define FTRAIN_CONSTRAINTS_HPP_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "flash_train/common.h"

namespace ftrain {

// Returns the calling thread's current device. A platform runtime failure
// throws Exception with FTRAIN_STATUS_INTERNAL_ERROR.
FTrainDeviceId getCurrentDeviceId();

// One selection's runtime constraint: the device the selection runs on.
class Constraints final {
  public:
    explicit Constraints(FTrainDeviceId device_id);
    Constraints(FTrainDeviceId device_id, std::string arch) : device_id_(device_id), arch_(std::move(arch)) {}

    FTrainDeviceId getDeviceId() const noexcept { return device_id_; }
    const std::string& getArch() const noexcept { return arch_; }

    // Encodes every field that can change which Primitive applies into
    // integer tokens: selections with equal tokens must select the same
    // records in the same order, because a cache hit skips isApplicable
    // and the Finders. Any field added to this class that can change a
    // selection outcome must be encoded here.
    std::vector<std::uint64_t> getConstraintsKey() const {
        std::vector<std::uint64_t> key{static_cast<std::uint64_t>(static_cast<std::uint32_t>(device_id_)),
                                       static_cast<std::uint64_t>(arch_.size())};
        for (unsigned char c : arch_) { key.push_back(c); }
        return key;
    }

  private:
    FTrainDeviceId device_id_;
    std::string arch_;
};

}  // namespace ftrain

#endif
