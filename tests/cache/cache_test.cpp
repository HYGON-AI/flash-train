// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
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
static_assert(std::is_trivially_copyable_v<CacheKey>);

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

// White-box helpers for the copy-returning cache.
void expectPublished(const MemoryPrimitiveCache& cache, const CacheKey& key,
                     const std::vector<std::size_t>& positions) {
    const std::optional<std::vector<std::size_t>> found = cache.find(key);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, positions);
}

void expectAbsent(const MemoryPrimitiveCache& cache, const CacheKey& key) { EXPECT_FALSE(cache.find(key).has_value()); }

TEST(MemoryPrimitiveCacheTest, PublishesOnceAndNeverReplacesAnExactKey) {
    MemoryPrimitiveCache cache;
    const CacheKey key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{7}};

    expectAbsent(cache, key);
    cache.publish(key, std::vector<std::size_t>{0});
    cache.publish(key, std::vector<std::size_t>{1});

    EXPECT_EQ(cache.getSize(), 1);
    expectPublished(cache, key, std::vector<std::size_t>{0});
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { cache.publish(key, std::vector<std::size_t>{}); });
}

TEST(MemoryPrimitiveCacheTest, RejectsANonPositiveEntryLimit) {
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [] { MemoryPrimitiveCache cache{0}; });
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [] { MemoryPrimitiveCache cache{8, 0}; });
}

TEST(MemoryPrimitiveCacheTest, EvictsTheLeastHitEntryWhenFull) {
    MemoryPrimitiveCache cache{2, 1};
    const CacheKey first_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{1}};
    const CacheKey second_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{2}};
    const CacheKey third_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{3}};
    const std::vector<std::size_t> first_positions{0};
    const std::vector<std::size_t> second_positions{1};
    const std::vector<std::size_t> third_positions{2};

    cache.publish(first_key, first_positions);
    cache.publish(second_key, second_positions);
    for (int hit = 0; hit < 3; ++hit) { expectPublished(cache, first_key, first_positions); }
    expectPublished(cache, second_key, second_positions);

    // The cache is full: the newcomer replaces the least-hit entry.
    cache.publish(third_key, third_positions);
    EXPECT_EQ(cache.getSize(), 2);
    expectPublished(cache, first_key, first_positions);
    expectAbsent(cache, second_key);
    expectPublished(cache, third_key, third_positions);

    // Publishing an existing key at capacity neither replaces nor evicts.
    cache.publish(first_key, std::vector<std::size_t>{7});
    expectPublished(cache, first_key, first_positions);
    EXPECT_EQ(cache.getSize(), 2);
}

TEST(MemoryPrimitiveCacheTest, EvictsTheLeastHitBatchWhenFull) {
    MemoryPrimitiveCache cache{16, 1};
    std::vector<CacheKey> keys;
    keys.reserve(17);
    for (std::size_t index = 0; index < 17; ++index) {
        keys.emplace_back(std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{index});
    }

    for (std::size_t index = 0; index < 16; ++index) { cache.publish(keys[index], std::vector<std::size_t>{index}); }
    for (std::size_t index = 0; index < 16; ++index) {
        for (std::size_t hit = 0; hit < index; ++hit) { expectPublished(cache, keys[index], {index}); }
    }

    // The shard is full with hits 0..15: publishing the 17th key evicts
    // the batch (16 / 8) of the two least-hit entries and the shard runs
    // below the limit until the batch refills.
    cache.publish(keys[16], std::vector<std::size_t>{16});
    EXPECT_EQ(cache.getSize(), 15);
    expectAbsent(cache, keys[0]);
    expectAbsent(cache, keys[1]);
    for (std::size_t index = 2; index < 17; ++index) { expectPublished(cache, keys[index], {index}); }
}

TEST(MemoryPrimitiveCacheTest, PublishesAndCountsEntriesAcrossShards) {
    MemoryPrimitiveCache cache;
    for (std::size_t index = 0; index < 1000; ++index) {
        const CacheKey key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{index}};
        cache.publish(key, std::vector<std::size_t>{index % 7});
    }

    EXPECT_EQ(cache.getSize(), 1000);
    const CacheKey middle_key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{500}};
    expectPublished(cache, middle_key, std::vector<std::size_t>{500 % 7});
}

TEST(MemoryPrimitiveCacheTest, SupportsConcurrentPublicationAndLookup) {
    MemoryPrimitiveCache cache;
    constexpr std::size_t kNumThreads = 8;
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        threads.emplace_back([&, index] {
            const std::vector<std::size_t> positions{index};
            const CacheKey key{std::vector<std::uint64_t>{0}, std::vector<std::uint64_t>{index}};
            cache.publish(key, positions);
            expectPublished(cache, key, positions);
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
    cache.publish(key, std::vector<std::size_t>{0});

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                cache.publish(key, std::vector<std::size_t>{index});
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
    expectPublished(cache, key, std::vector<std::size_t>{0});
    EXPECT_EQ(cache.getSize(), 1);
}

}  // namespace
}  // namespace ftrain
