#ifndef FTRAIN_HANDLE_HPP_
#define FTRAIN_HANDLE_HPP_

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "flash_train/engine.hpp"

namespace ftrain {

// Process-wide registry mapping PatternKeys to engines. Entries persist for
// the process lifetime and cannot be replaced or removed. All methods are
// thread-safe.
class Handle final {
  public:
    // Registers ops_engine under its PatternKey. A null engine or an
    // already-registered key throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT;
    // an existing entry is never replaced.
    void registerOpsEngine(std::shared_ptr<OpsEngineBase> ops_engine);

    // Returns the engine registered for pattern_key, or an empty shared_ptr.
    std::shared_ptr<const OpsEngineBase> findOpsEngine(const PatternKey& pattern_key) const;

    std::size_t getNumOpsEngines() const;

  private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<PatternKey, std::shared_ptr<const OpsEngineBase>, PatternKeyHasher> ops_engines_;
};

// Returns the process-wide Handle with every built-in engine already
// registered. Thread-safe; the first call performs the registration.
Handle& getGlobalHandle();

}  // namespace ftrain

#endif
