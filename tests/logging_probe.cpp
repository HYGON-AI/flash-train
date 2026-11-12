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

int main(int argc, char** argv) {
    using namespace ftrain::logging;
    if (argc != 2) { return 2; }
    const std::string scenario = argv[1];
    if (scenario == "levels") {
        FTRAIN_LOG(kError, "Error marker");
        FTRAIN_LOG(kWarning, "Warning marker");
        FTRAIN_LOG(kInfo, "Info marker");
        FTRAIN_LOG(kDebug, "Debug marker");
    } else if (scenario == "prefix") {
        FTRAIN_LOG(kWarning, "Prefix format check");
    } else if (scenario == "once") {
        (void)getLevel();
        setenv("FTRAIN_LOG_LEVEL", "DEBUG", 1);
        FTRAIN_LOG(kDebug, "Changed level marker");
        FTRAIN_LOG(kWarning, "Retained level marker");
    } else if (scenario == "lazy") {
        int calls = 0;
        FTRAIN_LOG(kDebug, "Argument evaluation count: %d", ++calls);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "exceptions") {
        errno = EDOM;
        log(Level::kDebug, [](Buffer&) {
            errno = EINVAL;
            throw std::bad_alloc{};
        });
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
        log(Level::kError, [](Buffer& record) {
            record.append("Large message ");
            for (int i = 0; i < 1000000 && !record.full(); ++i) { record.append("中文\n"); }
        });
    } else {
        return 6;
    }
    return 0;
}
