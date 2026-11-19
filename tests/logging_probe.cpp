// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "flash_train/logging.hpp"

namespace {

// Custom-type probe: format() overload rendering through the scratch arena.
struct Order {
    int id;
    int items;
};

inline void format(ftrain::log::FormatBuffer& out, const Order& order) noexcept {
    errno = EINVAL;  // pollution probe: the macro's entry guard must restore it
    out.format("Order(id=%d, items=%d)", order.id, order.items);
}

// Laziness probe: counts evaluations.
struct Counting {
    int* calls;
};

inline void format(ftrain::log::FormatBuffer& out, const Counting& value) noexcept {
    ++*value.calls;
    out.format("counted");
}

// Exception probe: renders by throwing inside the format() customization.
struct Throwing {};
inline void format(ftrain::log::FormatBuffer&, const Throwing&) {
    errno = EINVAL;
    throw std::bad_alloc{};
}

// Overflow probe: renders a scratch-filling segment to exercise record truncation.
struct LargePayload {};
inline void format(ftrain::log::FormatBuffer& out, const LargePayload&) noexcept {
    out.format("Large %5000d", 1);
}

// UTF-8 boundary probe: a long complete-codepoint payload whose record limit
// lands mid-character, exercising the codepoint-backing truncation.
struct Utf8Payload {};
inline void format(ftrain::log::FormatBuffer& out, const Utf8Payload&) noexcept {
    out.format("%s",
        "中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中");
}

// Composition probe: multiple writes must concatenate into one argument.
struct Team {
    int n;
};

inline void format(ftrain::log::FormatBuffer& out, const Team& team) noexcept {
    for (int i = 0; i < team.n; ++i) { out.format(" [%d]", i); }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace ftrain::log;
    if (argc != 2) { return 2; }
    const std::string scenario = argv[1];
    if (scenario == "levels") {
        FTRAIN_LOG(LogLevel::kError, "Error marker");
        FTRAIN_LOG(LogLevel::kWarning, "Warning marker");
        FTRAIN_LOG(LogLevel::kInfo, "Info marker");
        FTRAIN_LOG(LogLevel::kDebug, "Debug marker");
        FTRAIN_LOG(LogLevel::kTrace, "Trace marker");
    } else if (scenario == "prefix") {
        FTRAIN_LOG(LogLevel::kWarning, "Prefix format check");
    } else if (scenario == "once") {
        // Any non-kOff gate check touches getLevel: a disabled kTrace line
        // triggers read-once initialization silently, via the macro alone.
        FTRAIN_LOG(LogLevel::kTrace, "init trigger");
        setenv("FTRAIN_LOG_LEVEL", "DEBUG", 1);
        FTRAIN_LOG(LogLevel::kDebug, "Changed level marker");
        FTRAIN_LOG(LogLevel::kWarning, "Retained level marker");
    } else if (scenario == "lazy") {
        int calls = 0;
        FTRAIN_LOG(LogLevel::kDebug, "Argument evaluation count: %d", ++calls);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "custom") {
        errno = EDOM;
        const Order order{7, 3};
        const Order other{9, 9};
        FTRAIN_LOG(LogLevel::kWarning, "transparent=%s ok", order);
        FTRAIN_LOG(LogLevel::kWarning, "mix=%s and=%s", order, other);
        FTRAIN_LOG(LogLevel::kWarning, "at level=%s", LogLevel::kWarning);
        const LogLevel runtime_level = LogLevel::kWarning;
        FTRAIN_LOG(runtime_level, "runtime level=%s", LogLevel::kInfo);
        if (errno != EDOM) { return 7; }
    } else if (scenario == "lazy_custom") {
        int calls = 0;
        const Counting value{&calls};
        FTRAIN_LOG(LogLevel::kDebug, "eval=%s", value);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "exceptions") {
        errno = EDOM;
        FTRAIN_LOG(LogLevel::kDebug, "Failing %s", Throwing{});
        if (errno != EDOM) { return 3; }
        FTRAIN_LOG(LogLevel::kError, "Logging failure did not escape");
    } else if (scenario == "threads" || scenario == "init_race") {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 50; ++i) { FTRAIN_LOG(LogLevel::kWarning, "Thread %d record %d completed", t, i); }
            });
        }
        for (auto& thread : threads) { thread.join(); }
    } else if (scenario == "escape") {
        FTRAIN_LOG(LogLevel::kWarning, "esc [%s]", "a\nb\rc\td\x80\xff");
    } else if (scenario == "printf_edge") {
        FTRAIN_LOG(LogLevel::kError, "lit 100%%");
        FTRAIN_LOG(LogLevel::kError, "wide [%10000d]", 1);
        FTRAIN_LOG(LogLevel::kError, "bad [%lc]", static_cast<unsigned int>(0xd800));
    } else if (scenario == "utf8_trunc") {
        FTRAIN_LOG(LogLevel::kError, "%s", Utf8Payload{});
    } else if (scenario == "compose") {
        FTRAIN_LOG(LogLevel::kInfo, "team:%s", Team{3});
    } else if (scenario == "write_failure") {
        const int fd = open("/dev/full", O_WRONLY);
        if (fd < 0 || dup2(fd, STDERR_FILENO) < 0) { return 4; }
        close(fd);
        errno = EDOM;
        FTRAIN_LOG(LogLevel::kError, "Write failure probe");
        if (errno != EDOM) { return 5; }
        std::puts("errno=preserved");
    } else if (scenario == "large") {
        FTRAIN_LOG(LogLevel::kError, "%s", LargePayload{});
    } else {
        return 6;
    }
    return 0;
}
