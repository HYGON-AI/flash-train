// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "flash_train/family/sparse_attn_bwd/family.hpp"

namespace ftrain {
namespace {

TEST(SparseAttnBwdEngineTest, ProcessWideEngineIsOneImmutableSingleton) {
    const SparseAttnBwdEngine* first  = nullptr;
    const SparseAttnBwdEngine* second = nullptr;
    std::thread first_thread([&] { first = &sparseAttnBwdEngine(); });
    std::thread second_thread([&] { second = &sparseAttnBwdEngine(); });
    first_thread.join();
    second_thread.join();

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first, &sparseAttnBwdEngine());
}

TEST(SparseAttnBwdEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    // Discover all registered Primitives; do not maintain a full name list.
    // Configuration-specific support boundaries belong to Primitive tests.
    const auto records = SparseAttnBwdFamily::makeRecords();
    ASSERT_FALSE(records.empty());
    std::set<std::string> names;
    std::string registered_primitives;
    for (const auto& record : records) {
        ASSERT_NE(record, nullptr);
        ASSERT_NE(record->getName(), nullptr);
        EXPECT_STRNE(record->getName(), "");
        EXPECT_TRUE(names.insert(record->getName()).second);
        const auto clone = record->clone();
        ASSERT_NE(clone, nullptr);
        ASSERT_NE(clone.get(), record.get());
        EXPECT_STREQ(clone->getName(), record->getName());
        registered_primitives += std::string(record->getName()) + "\n";
    }
    RecordProperty("family", "sparse_attn_bwd");
    RecordProperty("registered_primitives", registered_primitives);
}

}  // namespace
}  // namespace ftrain
