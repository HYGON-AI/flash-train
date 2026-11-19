// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <thread>

#include <unistd.h>

#include "flash_train/env.h"
#include "flash_train/logging.hpp"

namespace ftrain {
namespace log {
namespace detail {
namespace {

const char* levelName(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::kError: return "ERROR";
        case LogLevel::kWarning: return "WARNING";
        case LogLevel::kInfo: return "INFO";
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kTrace: return "TRACE";
        default: return "OFF";
    }
}

// Bounded, single-line UTF-8 writer (file-local: nothing outside this
// translation unit references it). The final record, including newline and
// terminator, fits in kCapacity bytes. Further writes after truncation do
// nothing. format()/vformat() accept printf syntax.
class LogBuffer final {
  public:
    static constexpr std::size_t kCapacity = 4096;
    void append(std::string_view text) noexcept;
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    void
    format(const char* format, ...) noexcept;
    void vformat(const char* format, va_list arguments) noexcept;
    bool full() const noexcept { return truncated_; }
    const char* data() const noexcept { return data_.data(); }
    std::size_t size() const noexcept { return size_; }

  private:
    void truncate() noexcept;
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
    bool truncated_{false};
};

void prefix(LogBuffer& record, LogLevel level) noexcept {
    char timestamp[48]    = "time_unavailable";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_r(&now, &local) != nullptr) {
        if (std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &local) == 0) {
            std::snprintf(timestamp, sizeof(timestamp), "time_unavailable");
        }
    }
    record.format("[%s][ftrain][pid=%ld][thread=%zu][%s] ", timestamp, static_cast<long>(::getpid()),
                  std::hash<std::thread::id>{}(std::this_thread::get_id()), levelName(level));
}

void writeRecord(const LogBuffer& record) noexcept {
    // FTRAIN_LOG's entry guard spans every path here; direct callers of the
    // detail internals must install their own LogEntryGuard first.
    try {
        // Function-local lifetime; no business locks or callbacks inside this lock.
        static std::mutex mutex;
        char line[LogBuffer::kCapacity];
        std::memcpy(line, record.data(), record.size());
        line[record.size()] = '\n';
        const std::lock_guard<std::mutex> lock(mutex);
        // One best-effort write. No retries, buffering policy or signal changes.
        const auto written = ::write(STDERR_FILENO, line, record.size() + 1);
        (void)written;
    }
    catch (...) {
    }
}


bool equalsIgnoreCase(std::string_view text, const char* expected) noexcept {
    if (text.size() != std::strlen(expected)) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(text[i])) != expected[i]) { return false; }
    }
    return true;
}

LogLevel initialize() noexcept {
    const char* value     = std::getenv(FTRAIN_LOG_LEVEL);
    std::string_view text = value == nullptr ? std::string_view{} : std::string_view{value};
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) { text.remove_prefix(1); }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) { text.remove_suffix(1); }
    LogLevel level = LogLevel::kWarning;
    bool valid  = text.empty();
    for (LogLevel candidate :
         {LogLevel::kOff, LogLevel::kError, LogLevel::kWarning, LogLevel::kInfo, LogLevel::kDebug, LogLevel::kTrace}) {
        if (equalsIgnoreCase(text, levelName(candidate))) {
            level = candidate;
            valid = true;
            break;
        }
    }
    if (!valid || level >= LogLevel::kInfo) {
        LogBuffer record;
        const LogLevel message_level = valid ? LogLevel::kInfo : LogLevel::kWarning;
        prefix(record, message_level);
        if (valid) {
            record.format("Log level: %s", levelName(level));
        } else {
            record.append("Unknown FTRAIN_LOG_LEVEL; using WARNING");
        }
        // Bypass getLevel/log: initialization must never recursively enter itself.
        writeRecord(record);
    }
    return level;
}

LogLevel getLevel() noexcept {
    static const LogLevel level = []() noexcept { return initialize(); }();
    return level;
}

// Return a valid UTF-8 scalar length; invalid/partial encodings consume one
// input byte and are replaced by '?'. Never read beyond the bounded view.
std::size_t utf8Length(std::string_view text) noexcept {
    const auto c  = static_cast<unsigned char>(text[0]);
    std::size_t n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (n == 0 || text.size() < n) { return 0; }
    for (std::size_t i = 1; i < n; ++i) {
        if ((static_cast<unsigned char>(text[i]) & 0xc0) != 0x80) { return 0; }
    }
    const auto second = static_cast<unsigned char>(text[1]);
    if ((c == 0xe0 && second < 0xa0) || (c == 0xed && second >= 0xa0) || (c == 0xf0 && second < 0x90) ||
        (c == 0xf4 && second >= 0x90)) {
        return 0;
    }
    return n;
}

}  // namespace

void LogBuffer::truncate() noexcept {
    if (truncated_) { return; }
    constexpr std::string_view marker{"...[truncated]"};
    constexpr std::size_t limit = kCapacity - 2 - marker.size();
    if (size_ > limit) {
        std::size_t boundary = limit;
        while (boundary > 0 && (static_cast<unsigned char>(data_[boundary]) & 0xc0) == 0x80) { --boundary; }
        size_ = boundary;
    }
    std::memcpy(data_.data() + size_, marker.data(), marker.size());
    size_        += marker.size();
    data_[size_]  = '\0';
    truncated_    = true;
}

void LogBuffer::append(std::string_view text) noexcept {
    while (!text.empty() && !truncated_) {
        const auto c = static_cast<unsigned char>(text.front());
        char escaped[5]{};
        const char* bytes    = text.data();
        std::size_t consumed = 1;
        std::size_t count    = 1;
        if (c == '\n' || c == '\r' || c == '\t') {
            escaped[0] = '\\';
            escaped[1] = c == '\n' ? 'n' : c == '\r' ? 'r' : 't';
            bytes      = escaped;
            count      = 2;
        } else if (c < 0x20 || c == 0x7f) {
            std::snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            bytes = escaped;
            count = 4;
        } else if (c >= 0x80) {
            consumed = utf8Length(text);
            if (consumed == 0) {
                consumed   = 1;
                escaped[0] = '?';
                bytes      = escaped;
            } else {
                count = consumed;
            }
        }
        if (size_ + count > kCapacity - 2) {
            truncate();
            break;
        }
        std::memcpy(data_.data() + size_, bytes, count);
        size_        += count;
        data_[size_]  = '\0';
        text.remove_prefix(consumed);
    }
}

void LogBuffer::format(const char* format, ...) noexcept {
    std::va_list arguments;
    va_start(arguments, format);
    vformat(format, arguments);
    va_end(arguments);
}

void LogBuffer::vformat(const char* format, va_list arguments) noexcept {
    if (full() || format == nullptr) { return; }
    char text[kCapacity]{};
    const int count = std::vsnprintf(text, sizeof(text), format, arguments);
    if (count < 0) {
        append("[format_error]");
        return;
    }
    append(std::string_view{text, std::min(static_cast<std::size_t>(count), sizeof(text) - 1)});
    if (static_cast<std::size_t>(count) >= sizeof(text)) { truncate(); }
}

bool shouldLog(LogLevel level) noexcept {
    if (level == LogLevel::kOff) { return false; }
    // Future filters (module, rate, sampling) extend this body; the macro shape never changes.
    return level <= getLevel();
}

FormatBuffer& scratch() noexcept {
    static thread_local FormatBuffer instance;
    return instance;
}

void outputLog(LogLevel level, const char* format, ...) noexcept {
    std::va_list arguments;
    va_start(arguments, format);
    LogBuffer record;
    prefix(record, level);
    record.vformat(format, arguments);
    va_end(arguments);
    writeRecord(record);
    // Reset after vformat: segments were read during formatting, and every
    // enabled entry ends here, so the next entry renders from offset zero.
    scratch().clear();
}

}  // namespace detail

std::string_view FormatBuffer::format(const char* format, ...) noexcept {
    std::string_view result{};
    if (format != nullptr && size_ < kCapacity) {
        std::va_list arguments;
        va_start(arguments, format);
        const int count = std::vsnprintf(data_.data() + size_, kCapacity - size_, format, arguments);
        va_end(arguments);
        if (count >= 0) {
            // vsnprintf leaves a pending terminator at data_[size_ + written]; the
            // next write overwrites it, so consecutive writes concatenate, and
            // seal() promotes the last one into the argument's terminator.
            const std::size_t written =
                std::min(static_cast<std::size_t>(count), kCapacity - size_ - 1);
            result = std::string_view{data_.data() + size_, written};
            size_ += written;
        } else {
            result = std::string_view{"[format_error]"};
        }
    }
    return result;
}
}  // namespace log

}  // namespace ftrain
