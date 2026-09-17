#ifndef FTRAIN_PLAN_HPP_
#define FTRAIN_PLAN_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/primitive/base.hpp"

namespace ftrain {

// One planned execution: the configured Primitive(s) selected for one call
// and the device they were configured on, ordered best-first -- primitive
// 0 is the selection's default. Every Primitive shares one workspace
// buffer.
class Plan final {
  public:
    // Takes ownership of primitive_operands and binds the plan to device_id.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when
    // primitive_operands is empty or holds a null Primitive. Allocation
    // failure throws std::bad_alloc.
    Plan(FTrainDeviceId device_id, std::vector<std::unique_ptr<PrimitiveBase>>&& primitive_operands);

    std::uint64_t getNumPrimitives() const noexcept { return primitives_.size(); }

    // Returns the workspace bytes primitive_index's execute() call needs.
    // An out-of-range index throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    std::uint64_t getPrimitiveRequiredWorkspaceBytes(std::uint64_t primitive_index) const;

    // Enqueues primitive_index's work on stream without synchronizing it.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when the index is
    // out of range or the calling thread's current device is not this
    // plan's device, or, per the Primitive, when workspace_bytes is below
    // getPrimitiveRequiredWorkspaceBytes() or the requirement is nonzero
    // while workspace is null. The Plan, the referenced memory, and the
    // workspace must stay valid until the enqueued work completes.
    void execute(std::uint64_t primitive_index, void* workspace, std::uint64_t workspace_bytes, FTrainStream stream);

  private:
    FTrainDeviceId device_id_;
    std::vector<std::unique_ptr<PrimitiveBase>> primitives_;
};

}  // namespace ftrain

#endif
