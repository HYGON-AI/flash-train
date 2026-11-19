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

// Thread-local, bounded arena for rendering custom types as log text.
// outputLog() resets it after each entry. Consecutive format() calls
// concatenate in order (each call's pending terminator is overwritten by the
// next write); seal() finalizes the accumulation into one terminated string
// that is safe as a "%s" argument for the duration of the entry. Overflow
// truncates. Lives in ftrain because format() overloads name it in their
// signatures.
class FormatBuffer final {
  public:
    static constexpr std::size_t kCapacity = 4096;
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    std::string_view format(const char* format, ...) noexcept;
    void clear() noexcept { size_ = 0; }
    std::size_t used() const noexcept { return size_; }
    const char* data() const noexcept { return data_.data(); }
    // Engine-facing: terminates the current accumulation and reserves the
    // slot so later writes cannot eat the terminator.
    void seal() noexcept {
        if (size_ < kCapacity) {
            data_[size_] = '\0';
            size_ += 1;
        }
    }

  private:
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
};

}  // namespace ftrain

namespace ftrain {
namespace detail {

// One FormatBuffer per thread; never races across threads.
FormatBuffer& scratch() noexcept;



template<typename Type, typename = void>
struct HasFormatOverload final : std::false_type {};
template<typename Type>
struct HasFormatOverload<Type, std::void_t<decltype(format(std::declval<FormatBuffer&>(), std::declval<const Type&>()))>>
    final : std::true_type {};

// Per-argument dispatch inside emitLog: custom types render first by writing
// into the scratch arena (multiple writes concatenate in order; the engine
// seals the terminator and returns this argument's start offset); builtins
// pass through untouched; anything else fails with a teaching message.
template<typename Type>
auto parseArg(const Type& value) noexcept {
    if constexpr (HasFormatOverload<Type>::value) {
        FormatBuffer& buf = scratch();
        const std::size_t begin = buf.used();
        try {
            format(buf, value);
        }
        catch (...) {
            return "[format_error]";
        }
        buf.seal();
        return buf.data() + begin;
    } else if constexpr (std::is_arithmetic_v<Type> || std::is_pointer_v<Type> ||
                         std::is_convertible_v<Type, const char*>) {
        return value;
    } else {
        static_assert(sizeof(Type) == 0,
                      "flash-train logging: type is not loggable. Overload "
                      "format(ftrain::FormatBuffer&, const T&) next to the type: write into "
                      "the scratch as often as needed (writes concatenate); the return value "
                      "is ignored");
    }
}

// Renders the prefix and the already-dispatched message, writes the record,
// then resets the scratch arena: it owns the arena's whole lifecycle, and the
// reset must stay after vformat because segments are read during it. Trusted
// caller: emitLog under the macro's entry guard; direct callers must install
// a LogEntryGuard first.
void outputLog(LogLevel level, const char* format, ...) noexcept;

// Guards a log entry against side effects on process state:
// Today errno is the only polluted resource;
// extend or compose here if more state ever needs preserving. The macro's
// expansion is its sole construction site and covers every internal path.
struct LogEntryGuard final {
    LogEntryGuard() noexcept : errno_(errno) {}
    ~LogEntryGuard() { errno = errno_; }

  private:
    int errno_;
};

// The gate FTRAIN_LOG checks before evaluating arguments.
bool shouldLog(LogLevel level) noexcept;

// Format-string entry behind the macro's gate: dispatches each argument
// through parseArg() and forwards to the printf pipeline. Note that forwarding
// through this template removes the format(printf) compile-time check on the
// format string; the macro's gate keeps argument evaluation lazy.
template<typename... Arguments>
void emitLog(LogLevel level, const char* fmt, const Arguments&... arguments) noexcept {
    try {
        outputLog(level, fmt, parseArg(arguments)...);
    }
    catch (...) {
        // Diagnostics are best effort. Never report a logging failure recursively.
    }
}

// ---- How to make a type loggable -------------------------------------------
// The customization point is the name `format`: overload it in your own
// namespace next to your type as
//   void format(ftrain::FormatBuffer& out, const MyType& value);
// writing into the buffer as many times as needed (writes concatenate in
// order), return value ignored; noexcept is optional - a throwing overload
// renders its argument as "[format_error]". The declaration below is the
// sample signature developers implement against. NOTE: it must stay inside detail
// BELOW the HasFormatOverload trait - ADL from FormatBuffer searches ftrain
// but never this nested namespace, and ordinary lookup only sees what is
// above the trait; either escape route makes every type look customized and
// breaks builtins with a link error.
template<typename Type>
void format(FormatBuffer& scratch, const Type& value);

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
