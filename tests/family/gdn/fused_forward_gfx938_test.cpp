// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "flash_train/primitive/hygon/gdn/fused_forward_gfx938.hpp"
#include "problem_cases.hpp"

namespace {

constexpr auto kKind = ftrain::OperationKind::kGdnForward;

TEST(GdnFusedForwardGfx938, GateAndCandidateNames) {
    const auto problem = ftrain_test::problem<kKind>(ftrain_test::ports<kKind>());
    const ftrain::Constraints gfx936{0, "gfx936"}, gfx938{0, "gfx938"};
    const ftrain::GdnFusedForwardGfx938 bf16{FTRAIN_NUMERIC_TYPE_BF16};
    const ftrain::GdnFusedForwardGfx938 fp32{FTRAIN_NUMERIC_TYPE_FP32};
    EXPECT_STREQ(bf16.getName(), "GdnFusedForwardBf16Gfx938");
    EXPECT_STREQ(fp32.getName(), "GdnFusedForwardFp32Gfx938");
    EXPECT_TRUE(bf16.isApplicable(problem, gfx938).isSuccess());
    EXPECT_FALSE(fp32.isApplicable(problem, gfx938).isSuccess());
    EXPECT_FALSE(bf16.isApplicable(problem, gfx936).isSuccess());
    EXPECT_EQ(bf16.getRequiredWorkspaceBytes(), 0U);
}

TEST(GdnFusedForwardGfx938, DtypesStateAndLayout) {
    const ftrain::GdnFusedForwardGfx938 bf16{FTRAIN_NUMERIC_TYPE_BF16};
    const ftrain::Constraints gfx938{0, "gfx938"};
    auto ports = ftrain_test::ports<kKind>();
    ports[5].dims = {2, 4, 8, 16};
    ports[7].dims = {2, 4, 8, 16};
    ports[5].memory = reinterpret_cast<void*>(0x1000);
    ports[7].memory = reinterpret_cast<void*>(0x1000);
    EXPECT_TRUE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    ports[5].memory = nullptr;
    EXPECT_FALSE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    ports[5].memory = reinterpret_cast<void*>(0x1000);
    ports[3].type = FTRAIN_NUMERIC_TYPE_BF16;
    EXPECT_FALSE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    ports[3].type = FTRAIN_NUMERIC_TYPE_FP32;
    ports[2].index = FTRAIN_INDEX_TYPE_INVALID;
    ports[2].strides = {4096, 64, 16, 2};
    EXPECT_FALSE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
}

TEST(GdnFusedForwardGfx938, ShapeGate) {
    const ftrain::GdnFusedForwardGfx938 bf16{FTRAIN_NUMERIC_TYPE_BF16};
    const ftrain::Constraints gfx938{0, "gfx938"};
    // K > 128 stays unsupported.
    auto ports = ftrain_test::ports<kKind>();
    ports[0].dims[3] = ports[1].dims[3] = 129;
    EXPECT_FALSE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    // Unlike the gfx936 candidate, any S >= 1 is covered by the padded route.
    ports = ftrain_test::ports<kKind>();
    for (std::size_t i = 0; i < ports.size(); ++i) {
        if (i != 5 && i != 7) ports[i].dims[1] = 65;
    }
    EXPECT_TRUE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    // V > 256 stays unsupported.
    ports = ftrain_test::ports<kKind>();
    for (auto& port : ports) {
        if (port.dims.size() == 4) port.dims[3] = 320;
    }
    EXPECT_FALSE(bf16.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
}

}  // namespace
