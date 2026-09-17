#ifndef FTRAIN_RESOURCES_HPP_
#define FTRAIN_RESOURCES_HPP_

#include <cstdint>

#include "flash_train/common.h"

namespace ftrain {

// The execution resources one Primitive dispatch consumes: the shared
// workspace buffer with its byte size, and the stream the work is
// enqueued on. Future execution resources join this class, keeping the
// Primitive interface stable.
class Resources final {
  public:
    Resources(void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) noexcept
        : workspace_(workspace), workspace_bytes_(workspace_bytes), stream_(stream) {}

    // Returns the shared workspace buffer; may be null when the consumer
    // requires no workspace bytes.
    void* getWorkspace() const noexcept { return workspace_; }

    // Returns the workspace buffer's size in bytes.
    std::uint64_t getWorkspaceBytes() const noexcept { return workspace_bytes_; }

    // Returns the stream the work is enqueued on.
    FTrainStream getStream() const noexcept { return stream_; }

  private:
    void* workspace_;
    std::uint64_t workspace_bytes_;
    FTrainStream stream_;
};

}  // namespace ftrain

#endif
