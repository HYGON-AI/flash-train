// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_CACHE_HPP_
#define FTRAIN_CACHE_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ftrain {

// The cache key for one Primitive selection: a 128-bit summary of the
// Constraints tokens plus the Problem tokens. The two lanes are mixed by
// different functions over both token groups -- each group prefixed by
// its token count, so different group boundaries never collide -- and
// two keys are equal exactly when both lanes match. At the default entry
// limit the chance of two different token sequences summarizing to one
// key is about 1e-24, so summary equality stands in for token equality.
class CacheKey final {
  public:
    CacheKey(std::vector<std::uint64_t>&& constraints_tokens, std::vector<std::uint64_t>&& problem_tokens) noexcept;

    bool operator==(const CacheKey& other) const noexcept;

    bool operator!=(const CacheKey& other) const noexcept { return !(*this == other); }

    std::size_t getHash() const noexcept { return static_cast<std::size_t>(lane_a_); }

  private:
    std::uint64_t lane_a_;
    std::uint64_t lane_b_;
};

class CacheKeyHasher final {
  public:
    std::size_t operator()(const CacheKey& key) const noexcept { return key.getHash(); }
};

// The in-memory selection cache: maps one CacheKey to the selected
// records' positions in the owning OpsEngine's records_, in preference
// order. find() counts one hit per entry. The entries live in one of
// shard_count independently locked maps (the default 256), and the entry
// limit is split evenly across them. Once a shard is full, publishing a
// new key evicts from that shard a batch of its least-hit entries, so a
// shard briefly runs below its share until the batch refills. Safe for
// concurrent calls, and publish() keeps the first positions published
// for a key: a later publish for the same key does not replace them.
class MemoryPrimitiveCache final {
  public:
    // Builds a cache holding the default entry limit -- 30,000,000
    // entries, roughly 3 GB at full occupancy -- across the default
    // shard count.
    MemoryPrimitiveCache();

    // A limit of zero throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    explicit MemoryPrimitiveCache(std::size_t max_entries);

    // A limit or shard count of zero throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT. Pinning the shard count to 1 makes
    // eviction observable deterministically.
    MemoryPrimitiveCache(std::size_t max_entries, std::size_t shard_count);

    // Returns the published positions for an exact key in published
    // order -- counting one hit for the entry -- or an empty optional on
    // a miss. The copy is taken under the shard's lock, so it stays
    // valid regardless of later evictions.
    std::optional<std::vector<std::size_t>> find(const CacheKey& key) const;

    // Publishes positions only when key is absent. An existing mapping
    // is retained. An empty list throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT. When the key's shard is full, a
    // batch of that shard's least-hit entries is evicted to make room.
    void publish(const CacheKey& key, std::vector<std::size_t> positions);

    // Returns the number of stored keys, summed across shards.
    std::size_t getSize() const;

  private:
    static constexpr std::size_t kDefaultMaxEntries    = 30000000;
    static constexpr std::size_t kDefaultShardCount    = 256;
    static constexpr std::size_t kEvictionBatchDivisor = 8;

    // One published selection plus its hit count.
    struct Entry {
        Entry() = default;

        explicit Entry(std::vector<std::size_t> entry_positions) : positions(std::move(entry_positions)) {}

        Entry(Entry&& other) noexcept : positions(std::move(other.positions)), hits(other.hits.load()) {}

        Entry& operator=(Entry&& other) noexcept {
            positions = std::move(other.positions);
            hits.store(other.hits.load(), std::memory_order_relaxed);
            return *this;
        }

        std::vector<std::size_t> positions;
        mutable std::atomic<std::uint64_t> hits{0};
    };

    using Selections = std::unordered_map<CacheKey, Entry, CacheKeyHasher>;

    // One independently locked share of the entry limit.
    struct Shard {
        explicit Shard(std::size_t entry_limit) : limit(entry_limit) {}

        mutable std::shared_mutex mutex;
        Selections selections;
        std::size_t limit;
    };

    // Evicts a batch of the shard's least-hit entries; the caller holds
    // the shard's exclusive lock. One batch amortizes the scan across
    // the batch's worth of publishes.
    void evictLeastHitBatch(Shard& shard);

    Shard& shardFor(const CacheKey& key);
    const Shard& shardFor(const CacheKey& key) const;

    // Shards are held by pointer so the vector can grow without moving
    // the mutexes.
    std::vector<std::unique_ptr<Shard>> shards_;
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
