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

constexpr Level kAllLevels[] = {Level::kOff,  Level::kError, Level::kWarning,
                                Level::kInfo, Level::kDebug, Level::kTrace};
static_assert(std::size(kAllLevels) == static_cast<std::size_t>(Level::kTrace) + 1);

bool permitImpl(Level level, Level gate) noexcept { return level != Level::kOff && level <= gate; }

bool ignoreCaseCompare(std::string_view text, const char* expected) noexcept {
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
        for (Level candidate : kAllLevels) {
            if (ignoreCaseCompare(text, levelToText(candidate))) {
                parsed = candidate;
                valid  = true;
                break;
            }
        }
        const Level my_level = valid ? Level::kInfo : Level::kWarning;
        // Every entry's flow - guard, gate, emit - but gated against the parsed
        // local: permit() reads this static, which is mid-initialization here.
        const EntryGuard guard;
        if (permitImpl(my_level, parsed)) {
            if (valid) {
                emit(my_level, "Log level: %s", levelToText(parsed));
            } else {
                emit(my_level, "Unknown FTRAIN_LOG_LEVEL; using WARNING");
            }
        }
        return parsed;
    }();
    return level;
}

bool permit(Level level) noexcept {
    // Future filters (module, rate, sampling) extend this body; the macro shape never changes.
    return permitImpl(level, getLevel());
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

void addPrefix(Buffer& buf, Level level) noexcept {
    char timestamp[48]    = "time_unavailable";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_r(&now, &local) != nullptr) {
        if (std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &local) == 0) {
            std::snprintf(timestamp, sizeof(timestamp), "time_unavailable");
        }
    }
    buf.appendFormat("[%s][ftrain][pid=%ld][thread=%zu][%s] ", timestamp, static_cast<long>(::getpid()),
                     std::hash<std::thread::id>{}(std::this_thread::get_id()), levelToText(level));
}

// Wire pass: renders raw input as one bounded line of valid UTF-8. \n\r\t
// become their escapes, other control bytes print as \xNN, invalid sequences
// become '?'. Cuts land between units, so the single trailing truncation
// marker never splits a codepoint.
void toOneLine(Buffer& out, const Buffer& input) {
    constexpr std::string_view marker{"...[truncated]"};
    constexpr std::size_t limit = Buffer::kCapacity - 2 - marker.size();
    const std::string_view text{input.data(), input.size()};
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        char escaped[5];
        const char* bytes    = text.data() + i;
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
            consumed = utf8Length(text.substr(i));
            if (consumed == 0) {
                consumed   = 1;
                escaped[0] = '?';
                bytes      = escaped;
            } else {
                count = consumed;
            }
        }
        if (out.size() + count > limit) {
            out.appendStr(marker);
            return;
        }
        out.appendStr(std::string_view{bytes, count});
        i += consumed;
    }
}

void writeRecord(const Buffer& record) {
    // FTRAIN_LOG's entry guard spans every path here; direct callers of the
    // detail internals must install their own EntryGuard first.
    try {
        // Function-local lifetime; no business locks or callbacks inside this lock.
        static std::mutex mutex;
        char line[Buffer::kCapacity];
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
        if (count < 0) {
            appendStr(std::string_view{"[format_error]"});
        } else {
            size_ += std::min(static_cast<std::size_t>(count), kCapacity - size_ - 1);
        }
    }
}

void Buffer::appendStr(std::string_view str) noexcept {
    if (str.empty() || size_ >= kCapacity - 1) { return; }
    const std::size_t room  = kCapacity - 1 - size_;
    const std::size_t count = str.size() < room ? str.size() : room;
    std::memcpy(data_.data() + size_, str.data(), count);
    size_        += count;
    data_[size_]  = '\0';
}

}  // namespace log
}  // namespace ftrain
