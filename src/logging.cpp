// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

namespace ftrain::logging {
namespace {

const char* levelName(Level level) noexcept {
    switch (level) {
        case Level::kError: return "ERROR";
        case Level::kWarning: return "WARNING";
        case Level::kInfo: return "INFO";
        case Level::kDebug: return "DEBUG";
        default: return "OFF";
    }
}

bool equalsIgnoreCase(std::string_view text, const char* expected) noexcept {
    if (text.size() != std::strlen(expected)) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(text[i])) != expected[i]) { return false; }
    }
    return true;
}

Level initialize() noexcept {
    try {
        const char* value     = std::getenv(FTRAIN_LOG_LEVEL);
        std::string_view text = value == nullptr ? std::string_view{} : std::string_view{value};
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) { text.remove_prefix(1); }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) { text.remove_suffix(1); }
        Level level = Level::kWarning;
        bool valid  = text.empty();
        for (Level candidate : {Level::kOff, Level::kError, Level::kWarning, Level::kInfo, Level::kDebug}) {
            if (equalsIgnoreCase(text, levelName(candidate))) {
                level = candidate;
                valid = true;
                break;
            }
        }
        if (!valid || level >= Level::kInfo) {
            Buffer record;
            prefix(record, valid ? Level::kInfo : Level::kWarning, "logging");
            record.format("event=config variable=FTRAIN_LOG_LEVEL level=%s%s", levelName(level),
                          valid ? "" : " reason=invalid_value_using_default");
            // Bypass getLevel/log: initialization must never recursively enter itself.
            writeRecord(record);
        }
        return level;
    }
    catch (...) {
        return Level::kOff;
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

void Buffer::truncate() noexcept {
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

void Buffer::append(std::string_view text) noexcept {
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

void Buffer::format(const char* format, ...) noexcept {
    if (full() || format == nullptr) { return; }
    char text[kCapacity]{};
    std::va_list arguments;
    va_start(arguments, format);
    const int count = std::vsnprintf(text, sizeof(text), format, arguments);
    va_end(arguments);
    if (count < 0) {
        append("[format_error]");
        return;
    }
    append(std::string_view{text, std::min(static_cast<std::size_t>(count), sizeof(text) - 1)});
    if (static_cast<std::size_t>(count) >= sizeof(text)) { truncate(); }
}

Level getLevel() noexcept {
    static const Level level = []() noexcept {
        const ErrnoGuard guard;
        return initialize();
    }();
    return level;
}

void prefix(Buffer& record, Level level, const char* module) noexcept {
    char timestamp[48]    = "time_unavailable";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_r(&now, &local) != nullptr) {
        if (std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &local) == 0) {
            std::snprintf(timestamp, sizeof(timestamp), "time_unavailable");
        }
    }
    record.format("[%s][ftrain][%s][pid=%ld][thread=%zu][%s] ", timestamp, levelName(level),
                  static_cast<long>(::getpid()), std::hash<std::thread::id>{}(std::this_thread::get_id()),
                  module == nullptr ? "unknown" : module);
}

void writeRecord(const Buffer& record) noexcept {
    const ErrnoGuard guard;
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

}  // namespace ftrain::logging
