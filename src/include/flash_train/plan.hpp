#ifndef FTRAIN_PLAN_HPP_
#define FTRAIN_PLAN_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// One planned execution: the configured Primitive(s) selected for one call
// and the device they were configured on. Every Primitive shares one
// workspace buffer, so the plan's workspace requirement is the maximum of
// its Primitives' requirements.
class Plan final {
  public:
    // Takes ownership of primitive_operands and binds the plan to device_id.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when
    // primitive_operands is empty or holds a null Primitive. Allocation
    // failure throws std::bad_alloc.
    Plan(FTrainDeviceId device_id, std::vector<std::unique_ptr<PrimitiveBase>>&& primitive_operands);

    std::uint64_t getNumPrimitives() const noexcept { return primitives_.size(); }

    // Returns the workspace bytes one execute() call needs: the maximum of
    // the Primitives' getRequiredWorkspaceBytes() values.
    std::uint64_t getRequiredWorkspaceBytes() const noexcept;

    // Enqueues every Primitive's work on stream without synchronizing it.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when the calling
    // thread's current device is not this plan's device, or, per Primitive,
    // when workspace_bytes is below getRequiredWorkspaceBytes() or the
    // requirement is nonzero while workspace is null; Primitives already
    // enqueued when a later one rejects keep running. The Plan, the
    // referenced memory, and the workspace must stay valid until the
    // enqueued work completes.
    void execute(void* workspace, std::uint64_t workspace_bytes, FTrainStream stream);

  private:
    FTrainDeviceId device_id_;
    std::vector<std::unique_ptr<PrimitiveBase>> primitives_;
};

}  // namespace ftrain

#endif
