// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Fresh-process probe: configuration is never reset inside a production logger.
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"
#include "flash_train/api.hpp"
#include "flash_train/diagnostics.hpp"

// Test-process-only allocator fault: no injection hook or reset in production.
static std::atomic<bool> fail_next_allocation{false};
void* operator new(std::size_t size) {
    if (fail_next_allocation.exchange(false)) { throw std::bad_alloc{}; }
    if (void* memory = std::malloc(size == 0 ? 1 : size)) { return memory; }
    throw std::bad_alloc{};
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
std::atomic<int> summaries{0};
struct Plain {};
struct Named {
    static const char* diagnosticName() { return "named"; }
};
struct Summary {
    static void describeProblem(const int&, ftrain::logging::Buffer& record) {
        ++summaries;
        record.append("value=7");
    }
};
struct Both : Summary, Named {};
struct Throwing {
    static void describeProblem(const int&, ftrain::logging::Buffer&) {
        throw std::runtime_error("diagnostic failure");
    }
};
struct ThrowingName {
    static const char* diagnosticName() { throw 42; }
};
}  // namespace

int main(int argc, char** argv) {
    using namespace ftrain;
    using namespace ftrain::logging;
    if (argc != 2) { return 2; }
    const std::string scenario = argv[1];
    if (scenario == "initialization_failure") {
        { ScopedTraceRange warmup{"probe-warmup"}; }
        (void)getLevel();
        const FTrainStorageView empty{};
        fail_next_allocation = true;
        const auto first =
            ftrainGemm(empty, empty, empty, empty, empty, empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, nullptr);
        const auto second =
            ftrainGemm(empty, empty, empty, empty, empty, empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, nullptr);
        if (first != FTRAIN_STATUS_OUT_OF_MEMORY || second != first || fail_next_allocation.load() ||
            getLastResult().getStatus() != first || getLastResult().getMessage()[0] == '\0') {
            return 7;
        }
        std::puts("cached_initialization_failure=preserved");
    } else if (scenario == "levels") {
        FTRAIN_LOG(kError, "probe", "marker=error");
        FTRAIN_LOG(kWarning, "probe", "marker=warning");
        FTRAIN_LOG(kInfo, "probe", "marker=info");
        FTRAIN_LOG(kDebug, "probe", "marker=debug");
    } else if (scenario == "once") {
        (void)getLevel();
        setenv("FTRAIN_LOG_LEVEL", "DEBUG", 1);
        FTRAIN_LOG(kDebug, "probe", "marker=changed");
        FTRAIN_LOG(kWarning, "probe", "marker=retained");
    } else if (scenario == "lazy") {
        int calls = 0;
        FTRAIN_LOG(kDebug, "probe", "calls=%d", ++calls);
        std::printf("calls=%d\n", calls);
    } else if (scenario == "exceptions") {
        errno = EDOM;
        log(Level::kDebug, "probe", [](Buffer&) {
            errno = EINVAL;
            throw std::bad_alloc{};
        });
        if (errno != EDOM) { return 3; }
        FTRAIN_LOG(kError, "probe", "marker=survived");
    } else if (scenario == "diagnostics") {
        logProblem<Plain>(1);
        logProblem<Named>(1);
        logProblem<Summary>(1);
        logProblem<Both>(1);
        logProblem<Throwing>(1);
        logProblem<ThrowingName>(1);
        std::printf("summaries=%d\n", summaries.load());
    } else if (scenario == "threads" || scenario == "init_race") {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 50; ++i) {
                    FTRAIN_LOG(kWarning, "probe", "event=thread t=%d i=%d end=record", t, i);
                }
            });
        }
        for (auto& thread : threads) { thread.join(); }
    } else if (scenario == "api" || scenario == "write_failure") {
        if (scenario == "write_failure") {
            const int fd = open("/dev/full", O_WRONLY);
            if (fd < 0 || dup2(fd, STDERR_FILENO) < 0) { return 4; }
            close(fd);
        }
        errno = EDOM;
        const auto status =
            invokeApi("probeFailure", [] { throw Exception(FTRAIN_STATUS_UNSUPPORTED, "original failure"); });
        if (status != FTRAIN_STATUS_UNSUPPORTED || getLastResult().getStatus() != status ||
            std::string{getLastResult().getMessage()} != "original failure" || errno != EDOM) {
            return 5;
        }
        (void)ftrainGetLastStatus();
        (void)ftrainGetLastMessage();
        const Result failure = getLastResult();
        clearLastResult();
        setLastResult(failure.getStatus(), failure.getMessage());
        std::puts("result=preserved");
    } else if (scenario == "large") {
        log(Level::kError, "probe", [](Buffer& record) {
            record.append("event=large ");
            for (int i = 0; i < 1000000 && !record.full(); ++i) { record.append("中文\n"); }
        });
    } else {
        return 6;
    }
    return 0;
}
