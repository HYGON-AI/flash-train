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

constexpr std::uint64_t kSelectionHashOffset = 14695981039346656037ULL;
constexpr std::uint64_t kSelectionHashPrime  = 1099511628211ULL;

void mixSelectionHash(std::uint64_t operand, std::uint64_t& hash) noexcept {
    hash ^= operand;
    hash *= kSelectionHashPrime;
    hash ^= operand >> 32;
    hash *= kSelectionHashPrime;
}

std::size_t makeSelectionHash(const std::vector<std::uint64_t>& constraints_tokens,
                              const std::vector<std::uint64_t>& problem_tokens) noexcept {
    std::uint64_t hash = kSelectionHashOffset;
    for (const std::uint64_t token : constraints_tokens) { mixSelectionHash(token, hash); }
    for (const std::uint64_t token : problem_tokens) { mixSelectionHash(token, hash); }
    return static_cast<std::size_t>(hash);
}

}  // namespace

CacheKey::CacheKey(std::vector<std::uint64_t>&& constraints_tokens,
                   std::vector<std::uint64_t>&& problem_tokens) noexcept
    : constraints_tokens_(std::move(constraints_tokens)), problem_tokens_(std::move(problem_tokens)),
      hash_(makeSelectionHash(constraints_tokens_, problem_tokens_)) {}

CacheKey::CacheKey(CacheKey&& other) noexcept
    : constraints_tokens_(std::move(other.constraints_tokens_)), problem_tokens_(std::move(other.problem_tokens_)),
      hash_(makeSelectionHash(constraints_tokens_, problem_tokens_)) {
    other.hash_ = makeSelectionHash(other.constraints_tokens_, other.problem_tokens_);
}

CacheKey& CacheKey::operator=(CacheKey&& other) noexcept {
    if (this == &other) { return *this; }

    constraints_tokens_ = std::move(other.constraints_tokens_);
    problem_tokens_     = std::move(other.problem_tokens_);
    hash_               = makeSelectionHash(constraints_tokens_, problem_tokens_);
    other.hash_         = makeSelectionHash(other.constraints_tokens_, other.problem_tokens_);
    return *this;
}

bool CacheKey::operator==(const CacheKey& other) const noexcept {
    return constraints_tokens_ == other.constraints_tokens_ && problem_tokens_ == other.problem_tokens_;
}

MemoryPrimitiveCache::MemoryPrimitiveCache() : MemoryPrimitiveCache(kDefaultMaxEntries) {}

MemoryPrimitiveCache::MemoryPrimitiveCache(std::size_t max_entries) : max_entries_(max_entries) {
    if (max_entries == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MemoryPrimitiveCache requires a positive entry limit");
    }
}

std::shared_ptr<const std::vector<std::string>> MemoryPrimitiveCache::find(const CacheKey& key) const {
    const std::shared_lock lock(mutex_);
    const auto iterator = selections_.find(key);
    if (iterator == selections_.end()) { return {}; }
    iterator->second.hits.fetch_add(1, std::memory_order_relaxed);
    return iterator->second.names;
}

void MemoryPrimitiveCache::publish(const CacheKey& key, std::vector<std::string> names) {
    if (names.empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MemoryPrimitiveCache cannot publish an empty Primitive list");
    }
    const std::shared_ptr<const std::vector<std::string>> snapshot =
        std::make_shared<const std::vector<std::string>>(std::move(names));

    const std::unique_lock lock(mutex_);
    if (selections_.size() >= max_entries_ && selections_.find(key) == selections_.end()) { evictLeastHit(); }
    selections_.emplace(key, Entry{snapshot});
}

void MemoryPrimitiveCache::evictLeastHit() {
    const auto victim =
        std::min_element(selections_.begin(), selections_.end(), [](const auto& left, const auto& right) {
            return left.second.hits.load(std::memory_order_relaxed) < right.second.hits.load(std::memory_order_relaxed);
        });
    if (victim != selections_.end()) { selections_.erase(victim); }
}

std::size_t MemoryPrimitiveCache::getSize() const {
    const std::shared_lock lock(mutex_);
    return selections_.size();
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
