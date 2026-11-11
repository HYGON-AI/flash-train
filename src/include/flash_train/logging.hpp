// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_LOGGING_HPP_
#define FTRAIN_LOGGING_HPP_

#include <array>
#include <cerrno>
#include <cstddef>
#include <string_view>
#include <utility>

namespace ftrain::logging {

enum class Level { kOff, kError, kWarning, kInfo, kDebug };

// Bounded, single-line UTF-8 writer. The final record, including newline and
// terminator, fits in kCapacity bytes. Further writes after truncation do nothing.
// format() accepts printf syntax; arguments must match the format string.
class Buffer final {
  public:
    static constexpr std::size_t kCapacity = 4096;
    void append(std::string_view text) noexcept;
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    void
    format(const char* format, ...) noexcept;
    bool full() const noexcept { return truncated_; }
    const char* data() const noexcept { return data_.data(); }
    std::size_t size() const noexcept { return size_; }

  private:
    void truncate() noexcept;
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
    bool truncated_{false};
};

// Read once, at first use; only FTRAIN_LOG_LEVEL is read during initialization.
// These functions and the log entry preserve errno. No public C ABI is added.
Level getLevel() noexcept;
inline bool isEnabled(Level level) noexcept { return level != Level::kOff && level <= getLevel(); }
void prefix(Buffer& record, Level level) noexcept;
void writeRecord(const Buffer& record) noexcept;

// A separate lifetime guard also covers callback/argument evaluation, not just
// the final write. The callback must be read-only and must not reenter logging
// or business APIs. Business operations belong outside this callback.
class ErrnoGuard final {
  public:
    ErrnoGuard() noexcept : saved_(errno) {}
    ~ErrnoGuard() noexcept { errno = saved_; }

  private:
    int saved_;
};

template<typename Function>
void log(Level level, Function&& function) noexcept {
    if (!isEnabled(level)) { return; }
    const ErrnoGuard guard;
    try {
        Buffer record;
        prefix(record, level);
        std::forward<Function>(function)(record);
        writeRecord(record);
    }
    catch (...) {
        // Diagnostics are best effort. Never report a logging failure recursively.
    }
}

}  // namespace ftrain::logging

// The message format and its arguments are evaluated only after log() checks
// the level. Evaluation and output failures never escape into business code.
#define FTRAIN_LOG(level, ...)                                                                                   \
    ::ftrain::logging::log(::ftrain::logging::Level::level, [&](::ftrain::logging::Buffer& ftrain_log_buffer_) { \
        ftrain_log_buffer_.format(__VA_ARGS__);                                                                  \
    })

#endif
