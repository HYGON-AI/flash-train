// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectStatus(FTrainStatus status, Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "operation unexpectedly succeeded";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), status);
    }
}

// Minimal family policy for registry tests: engine construction only needs
// a Pattern and an empty record list; selection is never exercised.
struct RegistrationProblem {
    std::vector<std::uint64_t> getProblemKey() const { return {}; }
};

struct RegistrationFamily {
    using Problem = RegistrationProblem;
    using Records = std::vector<std::shared_ptr<const Primitive<Problem>>>;

    static Pattern makePattern() {
        PatternBuilder pattern;
        static_cast<void>(pattern.addOperand<OperandKind::kTensor>());
        return pattern.buildPattern();
    }

    static Records makeRecords() { return {}; }

    static Problem makeProblem(const Args&) { return {}; }
};

TEST(HandleTest, RegistersAndFindsExactOpsEnginesAndRejectsDuplicates) {
    Handle handle;
    const auto tensor_engine    = std::make_shared<OpsEngine<RegistrationFamily>>();
    const PatternKey tensor_key = tensor_engine->getPattern().getKey();

    EXPECT_EQ(handle.findOpsEngine(tensor_key), nullptr);
    handle.registerOpsEngine(tensor_engine);
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
    EXPECT_EQ(handle.findOpsEngine(tensor_key), tensor_engine);

    const auto duplicate_engine = std::make_shared<OpsEngine<RegistrationFamily>>();
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { handle.registerOpsEngine(duplicate_engine); });
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { handle.registerOpsEngine(nullptr); });
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
}

TEST(HandleTest, ConcurrentDuplicateRegistrationHasOneWinnerAndLookupRemainsSafe) {
    Handle handle;
    constexpr std::size_t kNumRegistrationThreads = 8;
    constexpr std::size_t kNumLookupThreads       = 4;
    std::vector<std::shared_ptr<OpsEngine<RegistrationFamily>>> engines;
    engines.reserve(kNumRegistrationThreads);
    for (std::size_t index = 0; index < kNumRegistrationThreads; ++index) {
        engines.push_back(std::make_shared<OpsEngine<RegistrationFamily>>());
    }
    const PatternKey key = engines.front()->getPattern().getKey();

    std::vector<FTrainStatus> statuses(kNumRegistrationThreads, FTRAIN_STATUS_INTERNAL_ERROR);
    std::vector<std::exception_ptr> errors(kNumRegistrationThreads);
    std::vector<std::thread> threads;
    std::atomic<std::size_t> ready{0};
    std::atomic<std::size_t> registrations_remaining{kNumRegistrationThreads};
    std::atomic<bool> start{false};
    std::atomic<bool> lookup_failed{false};
    threads.reserve(kNumRegistrationThreads + kNumLookupThreads);

    const auto is_candidate = [&](const std::shared_ptr<const OpsEngineBase>& found) {
        return std::any_of(engines.begin(), engines.end(), [&](const auto& engine) { return engine == found; });
    };
    for (std::size_t index = 0; index < kNumRegistrationThreads; ++index) {
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                handle.registerOpsEngine(engines[index]);
                statuses[index] = FTRAIN_STATUS_SUCCESS;
            }
            catch (const Exception& exception) {
                statuses[index] = exception.getResult().getStatus();
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
            registrations_remaining.fetch_sub(1, std::memory_order_release);
        });
    }
    for (std::size_t index = 0; index < kNumLookupThreads; ++index) {
        threads.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            while (registrations_remaining.load(std::memory_order_acquire) != 0) {
                const std::shared_ptr<const OpsEngineBase> found = handle.findOpsEngine(key);
                if ((found != nullptr && !is_candidate(found)) || handle.getNumOpsEngines() > 1) {
                    lookup_failed.store(true, std::memory_order_relaxed);
                }
                std::this_thread::yield();
            }
            for (std::size_t lookup = 0; lookup < 64; ++lookup) {
                const std::shared_ptr<const OpsEngineBase> found = handle.findOpsEngine(key);
                if (found == nullptr || !is_candidate(found) || handle.getNumOpsEngines() != 1) {
                    lookup_failed.store(true, std::memory_order_relaxed);
                }
            }
        });
    }

    const std::size_t num_threads = kNumRegistrationThreads + kNumLookupThreads;
    while (ready.load(std::memory_order_acquire) != num_threads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }

    for (const std::exception_ptr& error : errors) { EXPECT_FALSE(error); }
    EXPECT_EQ(std::count(statuses.begin(), statuses.end(), FTRAIN_STATUS_SUCCESS), 1);
    EXPECT_EQ(std::count(statuses.begin(), statuses.end(), FTRAIN_STATUS_INVALID_ARGUMENT),
              static_cast<std::ptrdiff_t>(kNumRegistrationThreads - 1));
    EXPECT_FALSE(lookup_failed.load(std::memory_order_relaxed));
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
    const std::shared_ptr<const OpsEngineBase> registered = handle.findOpsEngine(key);
    ASSERT_NE(registered, nullptr);
    EXPECT_TRUE(is_candidate(registered));
}

}  // namespace
}  // namespace ftrain
