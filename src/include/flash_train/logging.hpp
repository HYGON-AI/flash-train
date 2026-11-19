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

// Thread-local, bounded arena for rendering custom types as log text. emitLog()
// clears it at each entry; segments are NUL-terminated (safe as "%s"
// arguments) and stay valid until the next entry. Overflow truncates.
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

// Customization point for user types: provide an ADL-found overload next to
// the type, returning the rendered text, e.g.
//   std::string_view format(ftrain::detail::Scratch&, const MyType&) noexcept;
// It must be read-only, noexcept, and must not reenter logging or business
// APIs; a throwing overload renders as "[format_error]" instead of dropping
// the record.
template<typename Type, typename = void>
struct HasFormatOverload final : std::false_type {};
template<typename Type>
struct HasFormatOverload<Type, std::void_t<decltype(format(std::declval<Scratch&>(), std::declval<const Type&>()))>>
    final : std::true_type {};

// Per-argument dispatch inside emitLog: custom types render through their
// format() overload, builtins pass through, anything else fails at compile
// time with a readable message.
template<typename Type>
auto formatArg(const Type& value) noexcept {
    if constexpr (HasFormatOverload<Type>::value) {
        try {
            return format(scratch(), value).data();
        }
        catch (...) {
            return "[format_error]";
        }
    } else if constexpr (std::is_arithmetic_v<Type> || std::is_pointer_v<Type> ||
                         std::is_convertible_v<Type, const char*>) {
        return value;
    } else {
        static_assert(sizeof(Type) == 0,
                      "flash-train logging: type is not loggable; use a builtin, a format() "
                      "overload, or convert it before logging");
    }
}

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
