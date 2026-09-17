#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/selection.hpp"

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

std::size_t makeSelectionHash(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes,
                              const std::vector<std::uint64_t>& argument_tokens) noexcept {
    std::uint64_t hash = kSelectionHashOffset;
    mixSelectionHash(static_cast<std::uint64_t>(static_cast<std::uint32_t>(device_id)), hash);
    mixSelectionHash(max_workspace_bytes, hash);
    for (const std::uint64_t token : argument_tokens) { mixSelectionHash(token, hash); }
    return static_cast<std::size_t>(hash);
}

}  // namespace

SelectionKey::SelectionKey(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes,
                           std::vector<std::uint64_t>&& argument_tokens) noexcept
    : device_id_(device_id), max_workspace_bytes_(max_workspace_bytes), argument_tokens_(std::move(argument_tokens)),
      hash_(makeSelectionHash(device_id_, max_workspace_bytes_, argument_tokens_)) {}

SelectionKey::SelectionKey(SelectionKey&& other) noexcept
    : device_id_(other.device_id_), max_workspace_bytes_(other.max_workspace_bytes_),
      argument_tokens_(std::move(other.argument_tokens_)),
      hash_(makeSelectionHash(device_id_, max_workspace_bytes_, argument_tokens_)) {
    other.hash_ = makeSelectionHash(other.device_id_, other.max_workspace_bytes_, other.argument_tokens_);
}

SelectionKey& SelectionKey::operator=(SelectionKey&& other) noexcept {
    if (this == &other) { return *this; }

    device_id_           = other.device_id_;
    max_workspace_bytes_ = other.max_workspace_bytes_;
    argument_tokens_     = std::move(other.argument_tokens_);
    hash_                = makeSelectionHash(device_id_, max_workspace_bytes_, argument_tokens_);
    other.hash_          = makeSelectionHash(other.device_id_, other.max_workspace_bytes_, other.argument_tokens_);
    return *this;
}

bool SelectionKey::operator==(const SelectionKey& other) const noexcept {
    return device_id_ == other.device_id_ && max_workspace_bytes_ == other.max_workspace_bytes_ &&
           argument_tokens_ == other.argument_tokens_;
}

std::shared_ptr<const PrimitiveBase> MemoryPrimitiveCache::find(const SelectionKey& key) const {
    const std::shared_lock lock(mutex_);
    const auto iterator = records_.find(key);
    if (iterator == records_.end()) { return {}; }
    return iterator->second;
}

void MemoryPrimitiveCache::publish(const SelectionKey& key, std::shared_ptr<const PrimitiveBase> record) {
    if (record == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "PrimitiveCache cannot publish a null Primitive");
    }

    const std::unique_lock lock(mutex_);
    records_.emplace(key, std::move(record));
}

std::size_t MemoryPrimitiveCache::getSize() const {
    const std::shared_lock lock(mutex_);
    return records_.size();
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
    static const std::unordered_set<std::string> names = parseNameList(std::getenv("FTRAIN_ENABLED_PRIMITIVES"));
    return names;
}

const std::unordered_set<std::string>& getDisabledPrimitives() {
    static const std::unordered_set<std::string> names = parseNameList(std::getenv("FTRAIN_DISABLED_PRIMITIVES"));
    return names;
}

const std::unordered_set<std::string>& getDisabledFinders() {
    static const std::unordered_set<std::string> names = parseNameList(std::getenv("FTRAIN_DISABLED_FINDERS"));
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
    static const bool disabled = isSwitchEnabled(std::getenv("FTRAIN_DISABLE_SELECTION_CACHE"));
    return disabled;
}

bool enumeratesAllPrimitives() { return isSwitchEnabled(std::getenv("FTRAIN_ENUMERATE_ALL_PRIMITIVES")); }

}  // namespace ftrain
