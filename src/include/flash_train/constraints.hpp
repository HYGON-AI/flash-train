#ifndef FTRAIN_CONTEXT_HPP_
#define FTRAIN_CONTEXT_HPP_

#include <cstdint>

#include "flash_train/common.h"

namespace ftrain {

// Returns the calling thread's current device. A platform runtime failure
// throws Exception with FTRAIN_STATUS_INTERNAL_ERROR.
FTrainDeviceId getCurrentDeviceId();

// One selection's runtime constraints: the device and the maximum workspace
// bytes a selected Primitive may require.
class Constraints final {
  public:
    Constraints(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes) noexcept
        : device_id_(device_id), max_workspace_bytes_(max_workspace_bytes) {}

    FTrainDeviceId getDeviceId() const noexcept { return device_id_; }

    std::uint64_t getMaxWorkspaceBytes() const noexcept { return max_workspace_bytes_; }

  private:
    FTrainDeviceId device_id_;
    std::uint64_t max_workspace_bytes_;
};

}  // namespace ftrain

#endif
