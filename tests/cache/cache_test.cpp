// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/cache.hpp"

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

static_assert(noexcept(std::declval<const CacheKey&>().getHash()));
static_assert(std::is_nothrow_move_constructible_v<CacheKey>);
static_assert(std::is_nothrow_move_assignable_v<CacheKey>);

TEST(CacheKeyTest, ComparesBothTokenGroupsIndependently) {
    const CacheKey key{
        std::vector<std::uint64_t>{2},
        std::vector<std::uint64_t>{3, 5, 8}
    };
    const CacheKey same_key{
        std::vector<std::uint64_t>{2},
        std::vector<std::uint64_t>{3, 5, 8}
    };
    const CacheKey different_constraints{
        std::vector<std::uint64_t>{3},
        std::vector<std::uint64_t>{3, 5, 8}
    };
    const CacheKey different_problem{
        std::vector<std::uint64_t>{2},
        std::vector<std::uint64_t>{3, 5, 9}
    };
    // The same flattened tokens split differently across the two groups
    // stay distinct.
    const CacheKey shifted_boundary{
        std::vector<std::uint64_t>{2, 3},
        std::vector<std::uint64_t>{5, 8}
    };

    EXPECT_EQ(key, same_key);
    EXPECT_EQ(key.getHash(), same_key.getHash());
    EXPECT_NE(key, different_constraints);
    EXPECT_NE(key, different_problem);
    EXPECT_NE(key, shifted_boundary);
}

TEST(CacheKeyTest, PreservesHashInvariantAcrossMoves) {
    CacheKey first_key{
        std::vector<std::uint64_t>{2},
        std::vector<std::uint64_t>{3, 5, 8}
    };
    CacheKey second_key{
        std::vector<std::uint64_t>{2},
        std::vector<std::uint64_t>{13, 21}
    };
    const CacheKey expected_first_key  = first_key;
    const CacheKey expected_second_key = second_key;

    CacheKey moved_first_key(std::move(first_key));
    CacheKey moved_second_key = expected_first_key;
    moved_second_key          = std::move(second_key);

    EXPECT_EQ(moved_first_key, expected_first_key);
    EXPECT_EQ(moved_second_key, expected_second_key);
    if (first_key == second_key) { EXPECT_EQ(first_key.getHash(), second_key.getHash()); }
}

// White-box helpers for the snapshot-returning cache.
void expectPublished(const MemoryPrimitiveCache& cache, const CacheKey& key, const std::vector<std::string>& names) {
    const std::shared_ptr<const std::vector<std::string>> found = cache.find(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found, names);
}

void expectAbsent(const MemoryPrimitiveCache& cache, const CacheKey& key) { EXPECT_EQ(cache.find(key), nullptr); }

TEST(MemoryPrimitiveCacheTest, PublishesOnceAndNeverReplacesAnExactKey) {
    MemoryPrimitiveCache cache;
    const CacheKey key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{7}};

    expectAbsent(cache, key);
    cache.publish(key, std::vector<std::string>{"First"});
    cache.publish(key, std::vector<std::string>{"Second"});

    EXPECT_EQ(cache.getSize(), 1);
    expectPublished(cache, key, std::vector<std::string>{"First"});
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { cache.publish(key, std::vector<std::string>{}); });
}

TEST(MemoryPrimitiveCacheTest, RejectsANonPositiveEntryLimit) {
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [] { MemoryPrimitiveCache cache{0}; });
}

TEST(MemoryPrimitiveCacheTest, EvictsTheLeastHitEntryWhenFull) {
    MemoryPrimitiveCache cache{2};
    const CacheKey first_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{1}};
    const CacheKey second_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{2}};
    const CacheKey third_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{3}};
    const std::vector<std::string> first_names{"First"};
    const std::vector<std::string> second_names{"Second"};
    const std::vector<std::string> third_names{"Third"};

    cache.publish(first_key, first_names);
    cache.publish(second_key, second_names);
    for (int hit = 0; hit < 3; ++hit) { expectPublished(cache, first_key, first_names); }
    expectPublished(cache, second_key, second_names);

    // The cache is full: the newcomer replaces the least-hit entry.
    cache.publish(third_key, third_names);
    EXPECT_EQ(cache.getSize(), 2);
    expectPublished(cache, first_key, first_names);
    expectAbsent(cache, second_key);
    expectPublished(cache, third_key, third_names);

    // Publishing an existing key at capacity neither replaces nor evicts.
    cache.publish(first_key, std::vector<std::string>{"Other"});
    expectPublished(cache, first_key, first_names);
    EXPECT_EQ(cache.getSize(), 2);
}

TEST(MemoryPrimitiveCacheTest, SupportsConcurrentPublicationAndLookup) {
    MemoryPrimitiveCache cache;
    constexpr std::size_t kNumThreads = 8;
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        threads.emplace_back([&, index] {
            const std::vector<std::string> names{"TestPrimitive" + std::to_string(index)};
            const CacheKey key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{index}};
            cache.publish(key, names);
            expectPublished(cache, key, names);
        });
    }
    for (std::thread& thread : threads) { thread.join(); }

    EXPECT_EQ(cache.getSize(), kNumThreads);
}

TEST(MemoryPrimitiveCacheTest, ConcurrentExactKeyPublicationKeepsOneRecordWithoutReplacement) {
    MemoryPrimitiveCache cache;
    constexpr std::size_t kNumThreads = 8;
    const CacheKey key{
        std::vector<std::uint64_t>{0},
        std::vector<std::uint64_t>{13, 21}
    };
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(kNumThreads);
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    threads.reserve(kNumThreads);
    cache.publish(key, std::vector<std::string>{"Initial"});

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                cache.publish(key, std::vector<std::string>{"TestPrimitive" + std::to_string(index)});
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kNumThreads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }

    for (const std::exception_ptr& error : errors) { EXPECT_FALSE(error); }
    expectPublished(cache, key, std::vector<std::string>{"Initial"});
    EXPECT_EQ(cache.getSize(), 1);
}

}  // namespace
}  // namespace ftrain
