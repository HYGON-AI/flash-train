// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_LOGGING_HPP_
#define FTRAIN_LOGGING_HPP_

#include <array>
#include <cerrno>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ftrain {

// The level vocabulary of FTRAIN_LOG's first argument and of FTRAIN_LOG_LEVEL.
enum class LogLevel { kOff, kError, kWarning, kInfo, kDebug, kTrace };

namespace detail {

// Thread-local, bounded arena for rendering custom types as log text.
// outputLog() resets it after each entry; segments are NUL-terminated (safe
// as "%s" arguments) and stay valid for the duration of the entry. Overflow
// truncates.
class Scratch final {
  public:
    static constexpr std::size_t kCapacity = 4096;
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    std::string_view format(const char* format, ...) noexcept;
    void clear() noexcept { size_ = 0; }
    std::size_t used() const noexcept { return size_; }

  private:
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
};

// One Scratch per thread; never races across threads.
Scratch& scratch() noexcept;

// The customization contract, in code: overload this name next to your type
// (see SampleLoggable below for a working example); builtins never reach here
// because formatArg() short-circuits them.
template<typename Type>
auto format(Scratch& scratch, const Type& value) noexcept {
    static_assert(sizeof(Type) == 0,
                  "flash-train logging: overload format(ftrain::detail::Scratch&, const T&) "
                  "next to your type; return std::string_view (NUL-terminated; Scratch::format "
                  "guarantees it) or const char*");
}

// Per-argument dispatch inside emitLog: builtins pass through untouched;
// other types go through their format() overload (a missing one fails inside
// format() with a teaching message; a throwing one renders as an error).
template<typename Type>
auto formatArg(const Type& value) noexcept {
    if constexpr (std::is_arithmetic_v<Type> || std::is_pointer_v<Type> ||
                  std::is_convertible_v<Type, const char*>) {
        return value;
    } else {
        try {
            return format(scratch(), value).data();
        }
        catch (...) {
            return "[format_error]";
        }
    }
}

// Renders the prefix and the already-dispatched message, writes the record,
// then resets the scratch arena: it owns the arena's whole lifecycle, and the
// reset must stay after vformat because segments are read during it. Trusted
// caller: emitLog under the macro's entry guard; direct callers must install
// a LogEntryGuard first.
void outputLog(LogLevel level, const char* format, ...) noexcept;

// Guards a log entry against side effects on process state: snapshots errno
// for the whole record lifetime, including argument evaluation at the call
// site, not just the final write. Today errno is the only polluted resource;
// extend or compose here if more state ever needs preserving.
struct LogEntryGuard final {
    LogEntryGuard() noexcept : errno_(errno) {}
    ~LogEntryGuard() { errno = errno_; }

  private:
    int errno_;
};

// The gate FTRAIN_LOG checks before evaluating arguments.
bool shouldLog(LogLevel level) noexcept;

// Format-string entry behind the macro's gate: dispatches each argument
// through formatArg() and forwards to the printf pipeline. Note that forwarding
// through this template removes the format(printf) compile-time check on the
// format string; the macro's gate keeps argument evaluation lazy.
template<typename... Arguments>
void emitLog(LogLevel level, const char* fmt, const Arguments&... arguments) noexcept {
    try {
        outputLog(level, fmt, formatArg(arguments)...);
    }
    catch (...) {
        // Diagnostics are best effort. Never report a logging failure recursively.
    }
}

}  // namespace detail
}  // namespace ftrain

// The message format and its arguments are evaluated only after the gate
// checks the level; custom types with a format() overload may be passed
// directly where a "%s" appears. Evaluation and output failures never escape
// into business code. The "##" swallow drops the comma for calls with no
// arguments after the format string; the GNU-style extension is accepted by
// gcc/clang and by both MSVC preprocessors.
#define FTRAIN_LOG(level, fmt, ...)                                                   \
    do {                                                                              \
        const ::ftrain::detail::LogEntryGuard ftrain_log_guard_;                      \
        if (::ftrain::detail::shouldLog(::ftrain::LogLevel::level)) {                 \
            ::ftrain::detail::emitLog(::ftrain::LogLevel::level, fmt, ##__VA_ARGS__); \
        }                                                                             \
    } while (false)

#endif
