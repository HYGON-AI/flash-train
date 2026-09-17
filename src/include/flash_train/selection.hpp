#ifndef FTRAIN_SELECTION_HPP_
#define FTRAIN_SELECTION_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/primitive/base.hpp"

namespace ftrain {

// The cache key for one Primitive selection: device id, workspace byte limit,
// and the engine's argument tokens. Equality compares all three fields.
class SelectionKey final {
  public:
    SelectionKey(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes,
                 std::vector<std::uint64_t>&& argument_tokens) noexcept;
    SelectionKey(const SelectionKey&)            = default;
    SelectionKey& operator=(const SelectionKey&) = default;
    SelectionKey(SelectionKey&& other) noexcept;
    SelectionKey& operator=(SelectionKey&& other) noexcept;

    bool operator==(const SelectionKey& other) const noexcept;

    bool operator!=(const SelectionKey& other) const noexcept { return !(*this == other); }

    std::size_t getHash() const noexcept { return hash_; }

  private:
    FTrainDeviceId device_id_;
    std::uint64_t max_workspace_bytes_;
    std::vector<std::uint64_t> argument_tokens_;
    std::size_t hash_;
};

class SelectionKeyHasher final {
  public:
    std::size_t operator()(const SelectionKey& key) const noexcept { return key.getHash(); }
};

// A Primitive selection cache. Implementations must be safe for concurrent
// calls. publish() must keep the first record published for a key: a later
// publish for the same key must not replace it.
class PrimitiveCache {
  public:
    virtual ~PrimitiveCache() = default;

    // Returns the record stored for key, or an empty pointer on a miss.
    virtual std::shared_ptr<const PrimitiveBase> find(const SelectionKey& key) const = 0;

    virtual void publish(const SelectionKey& key, std::shared_ptr<const PrimitiveBase> record) = 0;
};

class MemoryPrimitiveCache final : public PrimitiveCache {
  public:
    // Returns the record for an exact key, or an empty shared_ptr on a miss.
    std::shared_ptr<const PrimitiveBase> find(const SelectionKey& key) const override;

    // Publishes record only when key is absent. An existing mapping is retained.
    // A null record throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    void publish(const SelectionKey& key, std::shared_ptr<const PrimitiveBase> record) override;

    // Returns the number of stored records.
    std::size_t getSize() const;

  private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<SelectionKey, std::shared_ptr<const PrimitiveBase>, SelectionKeyHasher> records_;
};

// Applies the operational name lists: when enabled is non-empty, name must be
// in it, and name must not be in disabled.
inline bool isPrimitiveAllowed(const std::string& name, const std::unordered_set<std::string>& enabled,
                               const std::unordered_set<std::string>& disabled) {
    if (!enabled.empty() && enabled.find(name) == enabled.end()) { return false; }
    return disabled.find(name) == disabled.end();
}

// Returns the process-wide enabled Primitive names, parsed once on the first
// call from the comma-separated FTRAIN_ENABLED_PRIMITIVES environment
// variable; the variable is not re-read later. Names are trimmed of
// surrounding spaces and tabs.
const std::unordered_set<std::string>& getEnabledPrimitives();

// Returns the process-wide disabled Primitive names, parsed once on the first
// call from the comma-separated FTRAIN_DISABLED_PRIMITIVES environment
// variable; the variable is not re-read later. Names are trimmed of
// surrounding spaces and tabs.
const std::unordered_set<std::string>& getDisabledPrimitives();

// Returns the process-wide disabled Finder names, parsed once on the first
// call from the comma-separated FTRAIN_DISABLED_FINDERS environment
// variable; the variable is not re-read later. Names are trimmed of
// surrounding spaces and tabs.
const std::unordered_set<std::string>& getDisabledFinders();

// Returns whether name appears in the disabled Finder list.
inline bool isFinderDisabled(const std::string& name) {
    const std::unordered_set<std::string>& disabled_finders = getDisabledFinders();
    return disabled_finders.find(name) != disabled_finders.end();
}

// Returns whether the selection cache is disabled: the
// FTRAIN_DISABLE_SELECTION_CACHE environment variable, read once on the
// first call, holds a trimmed value other than empty or "0".
bool isSelectionCacheDisabled();

}  // namespace ftrain

#endif
