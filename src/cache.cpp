// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include "flash_train/env.h"

#include "flash_train/error.hpp"
#include "flash_train/cache.hpp"

namespace ftrain {
namespace {

// Lane A: offset/prime mixing with a low-bit fold, one update per token.
constexpr std::uint64_t kLaneAOffset = 14695981039346656037ULL;
constexpr std::uint64_t kLaneAPrime  = 1099511628211ULL;

void mixLaneA(std::uint64_t operand, std::uint64_t& hash) noexcept {
    hash ^= operand;
    hash *= kLaneAPrime;
    hash ^= operand >> 32;
    hash *= kLaneAPrime;
}

// Lane B: a rotate-xor-shift scheme, structurally different from lane A
// so the two lanes fail independently.
constexpr std::uint64_t kLaneBGolden = 0x9E3779B97F4A7C15ULL;

void mixLaneB(std::uint64_t operand, std::uint64_t& hash) noexcept {
    hash ^= hash << 13;
    hash ^= hash >> 7;
    hash ^= hash << 17;
    hash += (operand + kLaneBGolden) * kLaneBGolden;
    hash ^= hash >> 29;
}

std::uint64_t makeLaneA(const std::vector<std::uint64_t>& constraints_tokens,
                        const std::vector<std::uint64_t>& problem_tokens) noexcept {
    std::uint64_t hash = kLaneAOffset;
    mixLaneA(static_cast<std::uint64_t>(constraints_tokens.size()), hash);
    for (const std::uint64_t token : constraints_tokens) { mixLaneA(token, hash); }
    mixLaneA(static_cast<std::uint64_t>(problem_tokens.size()), hash);
    for (const std::uint64_t token : problem_tokens) { mixLaneA(token, hash); }
    return hash;
}

std::uint64_t makeLaneB(const std::vector<std::uint64_t>& constraints_tokens,
                        const std::vector<std::uint64_t>& problem_tokens) noexcept {
    std::uint64_t hash = kLaneBGolden;
    mixLaneB(static_cast<std::uint64_t>(constraints_tokens.size()), hash);
    for (const std::uint64_t token : constraints_tokens) { mixLaneB(token, hash); }
    mixLaneB(static_cast<std::uint64_t>(problem_tokens.size()), hash);
    for (const std::uint64_t token : problem_tokens) { mixLaneB(token, hash); }
    return hash;
}

}  // namespace

CacheKey::CacheKey(std::vector<std::uint64_t>&& constraints_tokens,
                   std::vector<std::uint64_t>&& problem_tokens) noexcept
    : lane_a_(makeLaneA(constraints_tokens, problem_tokens)), lane_b_(makeLaneB(constraints_tokens, problem_tokens)) {}

bool CacheKey::operator==(const CacheKey& other) const noexcept {
    return lane_a_ == other.lane_a_ && lane_b_ == other.lane_b_;
}

MemoryPrimitiveCache::MemoryPrimitiveCache() : MemoryPrimitiveCache(kDefaultMaxEntries, kDefaultShardCount) {}

MemoryPrimitiveCache::MemoryPrimitiveCache(std::size_t max_entries)
    : MemoryPrimitiveCache(max_entries, kDefaultShardCount) {}

MemoryPrimitiveCache::MemoryPrimitiveCache(std::size_t max_entries, std::size_t shard_count) {
    if (max_entries == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MemoryPrimitiveCache requires a positive entry limit");
    }
    if (shard_count == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MemoryPrimitiveCache requires a positive shard count");
    }
    const std::size_t per_shard_limit = std::max<std::size_t>(1, (max_entries + shard_count - 1) / shard_count);
    shards_.reserve(shard_count);
    for (std::size_t index = 0; index < shard_count; ++index) {
        shards_.emplace_back(std::make_unique<Shard>(per_shard_limit));
    }
}

MemoryPrimitiveCache::Shard& MemoryPrimitiveCache::shardFor(const CacheKey& key) {
    return *shards_[key.getHash() % shards_.size()];
}

const MemoryPrimitiveCache::Shard& MemoryPrimitiveCache::shardFor(const CacheKey& key) const {
    return *shards_[key.getHash() % shards_.size()];
}

std::optional<std::vector<std::size_t>> MemoryPrimitiveCache::find(const CacheKey& key) const {
    const Shard& shard = shardFor(key);
    const std::shared_lock lock(shard.mutex);
    const auto iterator = shard.selections.find(key);
    if (iterator == shard.selections.end()) { return std::nullopt; }
    iterator->second.hits.fetch_add(1, std::memory_order_relaxed);
    return iterator->second.positions;
}

void MemoryPrimitiveCache::publish(const CacheKey& key, std::vector<std::size_t> positions) {
    if (positions.empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MemoryPrimitiveCache cannot publish an empty selection");
    }
    Shard& shard = shardFor(key);
    const std::unique_lock lock(shard.mutex);
    if (shard.selections.size() >= shard.limit && shard.selections.find(key) == shard.selections.end()) {
        evictLeastHitBatch(shard);
    }
    shard.selections.emplace(key, Entry{std::move(positions)});
}

void MemoryPrimitiveCache::evictLeastHitBatch(Shard& shard) {
    const std::size_t batch = std::max<std::size_t>(1, shard.selections.size() / kEvictionBatchDivisor);
    std::vector<std::pair<std::uint64_t, Selections::const_iterator>> candidates;
    candidates.reserve(shard.selections.size());
    for (auto iterator = shard.selections.begin(); iterator != shard.selections.end(); ++iterator) {
        candidates.emplace_back(iterator->second.hits.load(std::memory_order_relaxed), iterator);
    }
    std::nth_element(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(batch - 1), candidates.end(),
                     [](const auto& left, const auto& right) { return left.first < right.first; });
    for (std::size_t index = 0; index < batch; ++index) { shard.selections.erase(candidates[index].second); }
}

std::size_t MemoryPrimitiveCache::getSize() const {
    std::size_t size = 0;
    for (const std::unique_ptr<Shard>& shard : shards_) {
        const std::shared_lock lock(shard->mutex);
        size += shard->selections.size();
    }
    return size;
}

namespace {

// Splits a comma-separated environment variable value into trimmed names.
std::unordered_set<std::string> parseNameList(const char* value) {
    std::unordered_set<std::string> names;
    if (value == nullptr) { return names; }

    std::string::size_type begin = 0;
    const std::string text{value};
    while (begin <= text.size()) {
        const std::string::size_type end = text.find(',', begin);
        const std::string name = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::string::size_type first = name.find_first_not_of(" \t");
        const std::string::size_type last  = name.find_last_not_of(" \t");
        if (first != std::string::npos) { names.insert(name.substr(first, last - first + 1)); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return names;
}

}  // namespace

const std::unordered_set<std::string>& getEnabledPrimitives() {
    static const std::unordered_set<std::string> names = parseNameList(std::getenv(FTRAIN_ENABLED_PRIMITIVES));
    return names;
}

const std::unordered_set<std::string>& getDisabledPrimitives() {
    static const std::unordered_set<std::string> names = parseNameList(std::getenv(FTRAIN_DISABLED_PRIMITIVES));
    return names;
}

const std::unordered_set<std::string>& getDisabledFinders() {
    static const std::unordered_set<std::string> names = parseNameList(std::getenv(FTRAIN_DISABLED_FINDERS));
    return names;
}

// A switch variable counts as enabled when it holds a trimmed value other
// than empty or "0".
bool isSwitchEnabled(const char* value) noexcept {
    if (value == nullptr) { return false; }

    const std::string text{value};
    const std::string::size_type first = text.find_first_not_of(" \t");
    if (first == std::string::npos) { return false; }
    const std::string::size_type last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1) != "0";
}

bool isSelectionCacheDisabled() {
    static const bool disabled = isSwitchEnabled(std::getenv(FTRAIN_DISABLE_CACHE));
    return disabled;
}

}  // namespace ftrain
