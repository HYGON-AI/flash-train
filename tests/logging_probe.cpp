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

// Custom-type probe: toLogText overload rendering through the arena.
struct Order {
    int id;
    int items;
};

inline void toLogText(ftrain::log::Buffer& out, const Order& order) noexcept {
    errno = EINVAL;  // pollution probe: the macro's entry guard must restore it
    out.appendFormat("Order(id=%d, items=%d)", order.id, order.items);
}

// Laziness probe: counts evaluations.
struct Counting {
    int* calls;
};

inline void toLogText(ftrain::log::Buffer& out, const Counting& value) noexcept {
    ++*value.calls;
    out.appendFormat("counted");
}

// Exception probe: renders by throwing inside the format() customization.
struct Throwing {};
inline void toLogText(ftrain::log::Buffer&, const Throwing&) {
    errno = EINVAL;
    throw std::bad_alloc{};
}

// Overflow probe: renders a scratch-filling segment to exercise record truncation.
struct LargePayload {};
inline void toLogText(ftrain::log::Buffer& out, const LargePayload&) noexcept {
    out.appendFormat("Large %5000d", 1);
}

// UTF-8 boundary probe: a long complete-codepoint payload whose record limit
// lands mid-character, exercising the codepoint-backing truncation.
struct Utf8Payload {};
inline void toLogText(ftrain::log::Buffer& out, const Utf8Payload&) noexcept {
    out.appendFormat("%s",
        "中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中");
}

// Composition probe: multiple writes must concatenate into one argument.
struct Team {
    int n;
};

inline void toLogText(ftrain::log::Buffer& out, const Team& team) noexcept {
    for (int i = 0; i < team.n; ++i) { out.appendFormat(" [%d]", i); }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace ftrain::log;
    if (argc != 2) { return 2; }
    const std::string scenario = argv[1];
    if (scenario == "levels") {
        FTRAIN_LOG(Level::kError, "Error marker");
        FTRAIN_LOG(Level::kWarning, "Warning marker");
        FTRAIN_LOG(Level::kInfo, "Info marker");
        FTRAIN_LOG(Level::kDebug, "Debug marker");
        FTRAIN_LOG(Level::kTrace, "Trace marker");
    } else if (scenario == "prefix") {
        FTRAIN_LOG(Level::kWarning, "Prefix format check");
    } else if (scenario == "once") {
        // Any non-kOff gate check touches getLevel: a disabled kTrace line
        // triggers read-once initialization silently, via the macro alone.
        FTRAIN_LOG(Level::kTrace, "init trigger");
        setenv("FTRAIN_LOG_LEVEL", "DEBUG", 1);
        FTRAIN_LOG(Level::kDebug, "Changed level marker");
        FTRAIN_LOG(Level::kWarning, "Retained level marker");
    } else if (scenario == "lazy") {
        int calls = 0;
        FTRAIN_LOG(Level::kDebug, "Argument evaluation count: %d", ++calls);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "custom") {
        errno = EDOM;
        const Order order{7, 3};
        const Order other{9, 9};
        FTRAIN_LOG(Level::kWarning, "transparent=%s ok", order);
        FTRAIN_LOG(Level::kWarning, "mix=%s and=%s", order, other);
        FTRAIN_LOG(Level::kWarning, "at level=%s", Level::kWarning);
        const Level runtime_level = Level::kWarning;
        FTRAIN_LOG(runtime_level, "runtime level=%s", Level::kInfo);
        if (errno != EDOM) { return 7; }
    } else if (scenario == "lazy_custom") {
        int calls = 0;
        const Counting value{&calls};
        FTRAIN_LOG(Level::kDebug, "eval=%s", value);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "exceptions") {
        errno = EDOM;
        FTRAIN_LOG(Level::kDebug, "Failing %s", Throwing{});
        if (errno != EDOM) { return 3; }
        FTRAIN_LOG(Level::kError, "Logging failure did not escape");
    } else if (scenario == "threads" || scenario == "init_race") {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 50; ++i) { FTRAIN_LOG(Level::kWarning, "Thread %d record %d completed", t, i); }
            });
        }
        for (auto& thread : threads) { thread.join(); }
    } else if (scenario == "escape") {
        FTRAIN_LOG(Level::kWarning, "esc [%s]", "a\nb\rc\td\x80\xff");
    } else if (scenario == "printf_edge") {
        FTRAIN_LOG(Level::kError, "lit 100%%");
        FTRAIN_LOG(Level::kError, "wide [%10000d]", 1);
        FTRAIN_LOG(Level::kError, "bad [%lc]", static_cast<unsigned int>(0xd800));
    } else if (scenario == "utf8_trunc") {
        FTRAIN_LOG(Level::kError, "%s", Utf8Payload{});
    } else if (scenario == "compose") {
        FTRAIN_LOG(Level::kInfo, "team:%s", Team{3});
    } else if (scenario == "write_failure") {
        const int fd = open("/dev/full", O_WRONLY);
        if (fd < 0 || dup2(fd, STDERR_FILENO) < 0) { return 4; }
        close(fd);
        errno = EDOM;
        FTRAIN_LOG(Level::kError, "Write failure probe");
        if (errno != EDOM) { return 5; }
        std::puts("errno=preserved");
    } else if (scenario == "large") {
        FTRAIN_LOG(Level::kError, "%s", LargePayload{});
    } else {
        return 6;
    }
    return 0;
}
