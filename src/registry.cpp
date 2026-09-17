#include "flash_train/registry.hpp"

#include <mutex>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/gemm.hpp"

namespace ftrain {
namespace {

// Owns the process registry and completes built-in registration before the
// Handle can be observed by any caller. This is the composition root: the
// single sanctioned place where the engine layer references a concrete
// operator family.
class GlobalHandleState final {
  public:
    GlobalHandleState();

    Handle& getHandle() noexcept { return handle_; }

  private:
    Handle handle_;
};

GlobalHandleState::GlobalHandleState() { handle_.registerOpsEngine(makeGemmOpsEngine()); }

}  // namespace

void Handle::registerOpsEngine(std::shared_ptr<OpsEngineBase> ops_engine) {
    if (ops_engine == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine must not be null"); }

    const std::unique_lock lock(mutex_);
    const auto insertion = ops_engines_.emplace(ops_engine->getPatternKey(), std::move(ops_engine));
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
