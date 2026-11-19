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

// Logs to stderr when the level passes the FTRAIN_LOG_LEVEL threshold.
// Arguments are evaluated only after the level check; custom types render
// at "%s" through their toLogText() overload; evaluation and output failures
// never escape into business code.
#define FTRAIN_LOG(level, fmt, ...)                                                   \
    do {                                                                              \
        const ::ftrain::log::detail::EntryGuard ftrain_log_guard_;                   \
        const ::ftrain::log::Level ftrain_log_level_ = (level);                       \
        if (::ftrain::log::detail::permit(ftrain_log_level_)) {                       \
            ::ftrain::log::detail::emit(ftrain_log_level_, fmt, ##__VA_ARGS__);       \
        }                                                                             \
    } while (false)

namespace ftrain {
namespace log {

// The level vocabulary of FTRAIN_LOG's first argument and of FTRAIN_LOG_LEVEL.
enum class Level { kOff, kError, kWarning, kInfo, kDebug, kTrace };

// Bounded arena for rendering custom types as log text. Consecutive writes
// concatenate in order (each call's pending terminator is overwritten by the
// next write); ends() finalizes the accumulation into one terminated string
// that is safe as a "%s" argument. Overflow truncates.
struct Buffer final {
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    void
    appendFormat(const char* fmt, ...);
    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    const char* data() const noexcept { return data_.data(); }
    // Engine-facing: terminates the current accumulation and reserves the
    // slot so later writes cannot eat the terminator. Defined below.
    void ends() noexcept;

  private:
    static constexpr std::size_t kCapacity = 4096;
    std::array<char, kCapacity> data_{};
    std::size_t size_{0};
};

// How to make a type loggable:
// Overload `toLogText` in the same namespace as your type, writing into the
// buffer as many times as needed (writes concatenate in order); the return
// value is ignored and noexcept is optional - a throwing overload renders
// its argument as "[format_error]".
void toLogText(Buffer& out, Level level);

namespace detail {

// One Buffer per thread; never races across threads.
Buffer& scratch() noexcept;

template<typename Type, typename = void>
inline constexpr bool has_overload_v = false;
template<typename Type>
inline constexpr bool has_overload_v<Type, std::void_t<decltype(toLogText(std::declval<Buffer&>(), std::declval<const Type&>()))>> = true;

// Per-argument dispatch inside emit: custom types render first by writing
// into the scratch arena (multiple writes concatenate in order; the engine
// seals the terminator and returns this argument's start offset); builtins
// pass through untouched; anything else fails with a teaching message.
template<typename Type>
auto parseArg(const Type& value) noexcept {
    if constexpr (has_overload_v<Type>) {
        Buffer& buf = scratch();
        const std::size_t begin = buf.size();
        try {
            toLogText(buf, value);
        }
        catch (...) {
            return "[format_error]";
        }
        buf.ends();
        return buf.data() + begin;
    } else if constexpr (std::is_arithmetic_v<Type> || std::is_pointer_v<Type> ||
                         std::is_convertible_v<Type, const char*>) {
        return value;
    } else {
        static_assert(sizeof(Type) == 0,
                      "flash-train logging: type is not loggable. Overload "
                      "toLogText(ftrain::log::Buffer&, const T&) next to the type: write into "
                      "the buffer as often as needed (writes concatenate); the return value "
                      "is ignored");
    }
}

// Renders the prefix and the already-dispatched message, writes the record,
// then resets the scratch arena: it owns the arena's whole lifecycle, and the
// reset must stay after vformat because segments are read during it. Trusted
// caller: emit under the macro's entry guard; direct callers must install
// an EntryGuard first.
void output(Level level, const char* fmt, ...) noexcept;

// Guards a log entry against side effects on process state:
// Today errno is the only polluted resource;
// extend or compose here if more state ever needs preserving. The macro's
// expansion is its sole construction site and covers every internal path.
struct EntryGuard final {
    EntryGuard() noexcept : errno_(errno) {}
    ~EntryGuard() { errno = errno_; }

  private:
    int errno_;
};

// The gate FTRAIN_LOG checks before evaluating arguments.
bool permit(Level level) noexcept;

// Format-string entry behind the macro's gate: dispatches each argument
// through parseArg() and forwards to the printf pipeline. Note that forwarding
// through this template removes the format(printf) compile-time check on the
// fmt string; the macro's gate keeps argument evaluation lazy.
template<typename... Arguments>
void emit(Level level, const char* fmt, const Arguments&... arguments) noexcept {
    try {
        output(level, fmt, parseArg(arguments)...);
    }
    catch (...) {
        // Diagnostics are best effort. Never report a logging failure recursively.
    }
}

}  // namespace detail

// ==== public definitions =====================================================

inline void Buffer::ends() noexcept {
    if (size_ < kCapacity) {
        data_[size_] = '\0';
        size_ += 1;
    }
}

inline void toLogText(Buffer& out, Level level) {
    switch (level) {
        case Level::kOff: out.appendFormat("OFF"); break;
        case Level::kError: out.appendFormat("ERROR"); break;
        case Level::kWarning: out.appendFormat("WARNING"); break;
        case Level::kInfo: out.appendFormat("INFO"); break;
        case Level::kDebug: out.appendFormat("DEBUG"); break;
        case Level::kTrace: out.appendFormat("TRACE"); break;
    }
}

}  // namespace log
}  // namespace ftrain

#endif
