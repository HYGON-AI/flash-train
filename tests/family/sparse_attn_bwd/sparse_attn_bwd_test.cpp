// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/family/sparse_attn_bwd/family.hpp"

namespace ftrain {
namespace {

// A per-call engine would still pass every other test; only the shared
// selection cache would be lost.
TEST(SparseAttnBwdEngineTest, ReturnsTheSameEngineOnEveryCall) {
    EXPECT_EQ(&sparseAttnBwdEngine(), &sparseAttnBwdEngine());
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
