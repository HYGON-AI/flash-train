// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_CACHE_HPP_
#define FTRAIN_CACHE_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ftrain {

// The cache key for one Primitive selection: the Constraints tokens plus
// the Problem tokens, each encoded by its owning type and compared as
// separate groups. Equality compares both groups.
class CacheKey final {
  public:
    CacheKey(std::vector<std::uint64_t>&& constraints_tokens, std::vector<std::uint64_t>&& problem_tokens) noexcept;
    CacheKey(const CacheKey&)            = default;
    CacheKey& operator=(const CacheKey&) = default;
    CacheKey(CacheKey&& other) noexcept;
    CacheKey& operator=(CacheKey&& other) noexcept;

    bool operator==(const CacheKey& other) const noexcept;

    bool operator!=(const CacheKey& other) const noexcept { return !(*this == other); }

    std::size_t getHash() const noexcept { return hash_; }

  private:
    std::vector<std::uint64_t> constraints_tokens_;
    std::vector<std::uint64_t> problem_tokens_;
    std::size_t hash_;
};

class CacheKeyHasher final {
  public:
    std::size_t operator()(const CacheKey& key) const noexcept { return key.getHash(); }
};

// The in-memory selection cache: maps one CacheKey to the selected
// records' positions in the owning OpsEngine's records_, in preference
// order. find() counts one hit per entry.
// The cache holds at most its construction-time entry limit; once full,
// publishing a new key replaces one entry with the fewest hits. Safe for
// concurrent calls, and publish() keeps the first positions published for
// a key: a later publish for the same key does not replace them.
class MemoryPrimitiveCache final {
  public:
    // Builds a cache holding the default entry limit.
    MemoryPrimitiveCache();

    // A limit of zero throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    explicit MemoryPrimitiveCache(std::size_t max_entries);

    // Returns the published positions for an exact key in published order
    // -- counting one hit for the entry -- or a null pointer on a miss.
    // The returned snapshot shares the entry's immutable storage and stays
    // valid regardless of later evictions.
    std::shared_ptr<const std::vector<std::size_t>> find(const CacheKey& key) const;

    // Publishes positions only when key is absent. An existing mapping is
    // retained. An empty list throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT. When the cache is full, one entry
    // with the fewest hits is replaced to make room.
    void publish(const CacheKey& key, std::vector<std::size_t> positions);

    // Returns the number of stored keys.
    std::size_t getSize() const;

  private:
    static constexpr std::size_t kDefaultMaxEntries = 1024;

    // One published selection plus its hit count. positions is an
    // immutable snapshot so find() can hand it out without copying.
    struct Entry {
        Entry() = default;

        explicit Entry(std::shared_ptr<const std::vector<std::size_t>> entry_positions)
            : positions(std::move(entry_positions)) {}

        Entry(Entry&& other) noexcept : positions(std::move(other.positions)), hits(other.hits.load()) {}

        Entry& operator=(Entry&& other) noexcept {
            positions = std::move(other.positions);
            hits.store(other.hits.load(), std::memory_order_relaxed);
            return *this;
        }

        std::shared_ptr<const std::vector<std::size_t>> positions;
        mutable std::atomic<std::uint64_t> hits{0};
    };

    // Replaces one entry with the fewest hits; the caller holds the
    // exclusive lock.
    void evictLeastHit();

    std::size_t max_entries_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<CacheKey, Entry, CacheKeyHasher> selections_;
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

// Returns whether the selection cache is disabled: the FTRAIN_DISABLE_CACHE
// environment variable, read once on the first call, holds a trimmed value
// other than empty or "0".
bool isSelectionCacheDisabled();

}  // namespace ftrain

#endif
