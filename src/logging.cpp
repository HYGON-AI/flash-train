// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>

#include <unistd.h>

#include "flash_train/env.h"
#include "flash_train/logging.hpp"

namespace ftrain {
namespace log {
namespace detail {
namespace {

bool permitImpl(Level level, Level gate) noexcept { return level != Level::kOff && level <= gate; }

// Bounded, single-line UTF-8 writer (file-local: nothing outside this
// translation unit references it). The final record, including newline and
// terminator, fits in kCapacity bytes. Further writes after truncation do
// nothing. fmt()/vformat() accept printf syntax.
class LogBuffer final {
  public:
    static constexpr std::size_t kCapacity = 4096;
    void append(std::string_view text) noexcept;
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    void
    appendFormat(const char* fmt, ...) noexcept;
    void vformat(const char* fmt, va_list arguments) noexcept;
    bool full() const noexcept { return truncated_; }
    const char* data() const noexcept { return data_.data(); }
    std::size_t size() const noexcept { return size_; }

  private:
    void truncate() noexcept;
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
    bool truncated_{false};
};

bool equalsIgnoreCase(std::string_view text, const char* expected) noexcept {
    if (text.size() != std::strlen(expected)) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(text[i])) != expected[i]) { return false; }
    }
    return true;
}



Level getLevel() noexcept {
    static const Level level = []() noexcept {
        const char* value     = std::getenv(FTRAIN_LOG_LEVEL);
        std::string_view text = value == nullptr ? std::string_view{} : std::string_view{value};
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) { text.remove_prefix(1); }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) { text.remove_suffix(1); }
        Level parsed = Level::kWarning;
        bool valid   = text.empty();
        for (Level candidate :
             {Level::kOff, Level::kError, Level::kWarning, Level::kInfo, Level::kDebug, Level::kTrace}) {
            if (equalsIgnoreCase(text, levelToText(candidate))) {
                parsed = candidate;
                valid  = true;
                break;
            }
        }
        const Level banner_level = valid ? Level::kInfo : Level::kWarning;
        // Every entry's flow - guard, gate, emit - but gated against the parsed
        // local: permit() reads this static, which is mid-initialization here.
        const EntryGuard guard;
        if (permitImpl(banner_level, parsed)) {
            if (valid) {
                emit(banner_level, "Log level: %s", levelToText(parsed));
            } else {
                emit(banner_level, "Unknown FTRAIN_LOG_LEVEL; using WARNING");
            }
        }
        return parsed;
    }();
    return level;
}



void prefix(LogBuffer& record, Level level) noexcept {
    char timestamp[48]    = "time_unavailable";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_r(&now, &local) != nullptr) {
        if (std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &local) == 0) {
            std::snprintf(timestamp, sizeof(timestamp), "time_unavailable");
        }
    }
    record.appendFormat("[%s][ftrain][pid=%ld][thread=%zu][%s] ", timestamp, static_cast<long>(::getpid()),
                  std::hash<std::thread::id>{}(std::this_thread::get_id()), levelToText(level));
}

void writeRecord(const LogBuffer& record) noexcept {
    // FTRAIN_LOG's entry guard spans every path here; direct callers of the
    // detail internals must install their own EntryGuard first.
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

void LogBuffer::appendFormat(const char* fmt, ...) noexcept {
    std::va_list arguments;
    va_start(arguments, fmt);
    vformat(fmt, arguments);
    va_end(arguments);
}

void LogBuffer::vformat(const char* fmt, va_list arguments) noexcept {
    if (full() || fmt == nullptr) { return; }
    char text[kCapacity]{};
    const int count = std::vsnprintf(text, sizeof(text), fmt, arguments);
    if (count < 0) {
        append("[format_error]");
        return;
    }
    append(std::string_view{text, std::min(static_cast<std::size_t>(count), sizeof(text) - 1)});
    if (static_cast<std::size_t>(count) >= sizeof(text)) { truncate(); }
}

bool permit(Level level) noexcept {
    // Future filters (module, rate, sampling) extend this body; the macro shape never changes.
    return permitImpl(level, getLevel());
}

void output(Level level, const char* fmt, ...) noexcept {
    std::va_list arguments;
    va_start(arguments, fmt);
    LogBuffer record;
    prefix(record, level);
    record.vformat(fmt, arguments);
    va_end(arguments);
    writeRecord(record);
    // Reset after vformat: segments were read during formatting, and every
    // enabled entry ends here, so the next entry renders from offset zero.
    scratch().clear();
}

Buffer& scratch() noexcept {
    static thread_local Buffer instance;
    return instance;
}

}  // namespace detail

void Buffer::appendFormat(const char* fmt, ...) noexcept {
    if (fmt != nullptr && size_ < kCapacity) {
        std::va_list arguments;
        va_start(arguments, fmt);
        const int count = std::vsnprintf(data_.data() + size_, kCapacity - size_, fmt, arguments);
        va_end(arguments);
        // vsnprintf leaves a pending terminator at data_[size_ + written]; the
        // next write overwrites it, so consecutive writes concatenate, and
        // ends() promotes the last one into the argument's terminator.
        if (count >= 0) {
            size_ += std::min(static_cast<std::size_t>(count), kCapacity - size_ - 1);
        }
    }
}

void Buffer::appendChar(char c) noexcept {
    if (size_ < kCapacity - 1) {
        data_[size_]     = c;
        data_[size_ + 1] = '\0';
        size_           += 1;
    }
}

void Buffer::appendStr(const char* str) noexcept {
    if (str != nullptr) { appendStr(std::string_view{str}); }
}

void Buffer::appendStr(std::string_view str) noexcept {
    if (str.empty() || size_ >= kCapacity - 1) { return; }
    const std::size_t room  = kCapacity - 1 - size_;
    const std::size_t count = str.size() < room ? str.size() : room;
    std::memcpy(data_.data() + size_, str.data(), count);
    size_       += count;
    data_[size_] = '\0';
}
}  // namespace log

}  // namespace ftrain
