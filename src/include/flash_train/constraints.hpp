#ifndef FTRAIN_CONSTRAINTS_HPP_
#define FTRAIN_CONSTRAINTS_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"

namespace ftrain {

// Returns the calling thread's current device. A platform runtime failure
// throws Exception with FTRAIN_STATUS_INTERNAL_ERROR.
FTrainDeviceId getCurrentDeviceId();

// One selection's runtime constraint: the device the selection runs on.
class Constraints final {
  public:
    Constraints(FTrainDeviceId device_id) noexcept : device_id_(device_id) {}

    FTrainDeviceId getDeviceId() const noexcept { return device_id_; }

    // Encodes every field that can change which Primitive applies into
    // integer tokens: selections with equal tokens must select the same
    // records in the same order, because a cache hit skips isApplicable
    // and the Finders. Any field added to this class that can change a
    // selection outcome must be encoded here.
    std::vector<std::uint64_t> getConstraintsKey() const {
        return {static_cast<std::uint64_t>(static_cast<std::uint32_t>(device_id_))};
    }

  private:
    FTrainDeviceId device_id_;
};

}  // namespace ftrain

#endif
