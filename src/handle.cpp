#include <mutex>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

// Owns the process registry and completes built-in registration before the
// Handle can be observed by any caller. This is the composition root: it
// registers whatever the engine layer's built-in catalog provides and
// references no concrete engine family itself.
class GlobalHandleState final {
  public:
    GlobalHandleState();

    Handle& getHandle() noexcept { return handle_; }

  private:
    Handle handle_;
};

GlobalHandleState::GlobalHandleState() {
    for (std::shared_ptr<OpsEngineBase> ops_engine : makeBuiltinOpsEngines()) {
        handle_.registerOpsEngine(std::move(ops_engine));
    }
}

}  // namespace

void Handle::registerOpsEngine(std::shared_ptr<OpsEngineBase> ops_engine) {
    if (ops_engine == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine must not be null"); }

    const std::unique_lock lock(mutex_);
    const auto insertion = ops_engines_.emplace(ops_engine->getPattern().getKey(), std::move(ops_engine));
    if (!insertion.second) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "An OpsEngine is already registered for this PatternKey");
    }
}

std::shared_ptr<const OpsEngineBase> Handle::findOpsEngine(const PatternKey& pattern_key) const {
    const std::shared_lock lock(mutex_);
    const auto iterator = ops_engines_.find(pattern_key);
    if (iterator == ops_engines_.end()) { return {}; }
    return iterator->second;
}

std::size_t Handle::getNumOpsEngines() const {
    const std::shared_lock lock(mutex_);
    return ops_engines_.size();
}

Handle& getGlobalHandle() {
    static GlobalHandleState state;
    return state.getHandle();
}

}  // namespace ftrain
