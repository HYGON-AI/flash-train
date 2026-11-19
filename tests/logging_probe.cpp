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

inline std::string_view format(ftrain::detail::Scratch& out, const Order& order) noexcept {
    errno = EINVAL;  // pollution probe: the macro's entry guard must restore it
    return out.format("Order(id=%d, items=%d)", order.id, order.items);
}

// Laziness probe: counts evaluations.
struct Counting {
    int* calls;
};

inline std::string_view format(ftrain::detail::Scratch& out, const Counting& value) noexcept {
    ++*value.calls;
    return out.format("counted");
}

// Exception probe: renders by throwing inside the format() customization.
struct Throwing {};
inline std::string_view format(ftrain::detail::Scratch&, const Throwing&) {
    errno = EINVAL;
    throw std::bad_alloc{};
}

// Overflow probe: renders a scratch-filling segment to exercise record truncation.
struct LargePayload {};
inline std::string_view format(ftrain::detail::Scratch& out, const LargePayload&) noexcept {
    return out.format("Large %5000d", 1);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace ftrain;
    if (argc != 2) { return 2; }
    const std::string scenario = argv[1];
    if (scenario == "levels") {
        FTRAIN_LOG(kError, "Error marker");
        FTRAIN_LOG(kWarning, "Warning marker");
        FTRAIN_LOG(kInfo, "Info marker");
        FTRAIN_LOG(kDebug, "Debug marker");
        FTRAIN_LOG(kTrace, "Trace marker");
    } else if (scenario == "prefix") {
        FTRAIN_LOG(kWarning, "Prefix format check");
    } else if (scenario == "once") {
        (void)detail::getLevel();
        setenv("FTRAIN_LOG_LEVEL", "DEBUG", 1);
        FTRAIN_LOG(kDebug, "Changed level marker");
        FTRAIN_LOG(kWarning, "Retained level marker");
    } else if (scenario == "lazy") {
        int calls = 0;
        FTRAIN_LOG(kDebug, "Argument evaluation count: %d", ++calls);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "custom") {
        errno = EDOM;
        const Order order{7, 3};
        FTRAIN_LOG(kWarning, "transparent=%s ok", order);
        if (errno != EDOM) { return 7; }
    } else if (scenario == "lazy_custom") {
        int calls = 0;
        const Counting value{&calls};
        FTRAIN_LOG(kDebug, "eval=%s", value);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "exceptions") {
        errno = EDOM;
        FTRAIN_LOG(kDebug, "Failing %s", Throwing{});
        if (errno != EDOM) { return 3; }
        FTRAIN_LOG(kError, "Logging failure did not escape");
    } else if (scenario == "threads" || scenario == "init_race") {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 50; ++i) { FTRAIN_LOG(kWarning, "Thread %d record %d completed", t, i); }
            });
        }
        for (auto& thread : threads) { thread.join(); }
    } else if (scenario == "write_failure") {
        const int fd = open("/dev/full", O_WRONLY);
        if (fd < 0 || dup2(fd, STDERR_FILENO) < 0) { return 4; }
        close(fd);
        errno = EDOM;
        FTRAIN_LOG(kError, "Write failure probe");
        if (errno != EDOM) { return 5; }
        std::puts("errno=preserved");
    } else if (scenario == "large") {
        FTRAIN_LOG(kError, "%s", LargePayload{});
    } else {
        return 6;
    }
    return 0;
}
