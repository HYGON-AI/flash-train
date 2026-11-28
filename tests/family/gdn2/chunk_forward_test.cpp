// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "flash_train/primitive/hygon/gdn2/chunk_forward.hpp"
#include "../gdn/problem_cases.hpp"

namespace {

constexpr auto kKind = ftrain::OperationKind::kGdn2Forward;

std::vector<ftrain_test::Port> validPorts() {
    auto ports = ftrain_test::ports<kKind>();
    for (auto& port : ports) {
        port.dims[0] = 1;
        port.dims[1] = 512;
        port.dims.back() = 64;
    }
    ports[0].dims[2] = ports[1].dims[2] = 1;
    ports[2].dims[2] = ports[3].dims[2] = ports[4].dims[2] =
        ports[5].dims[2] = ports[6].dims[2] = 2;
    return ports;
}

TEST(Gdn2ChunkForward, ShapeDtypeAndArchGate) {
    const ftrain::Gdn2ChunkForward primitive;
    const ftrain::Constraints gfx936{0, "gfx936"}, gfx938{0, "gfx938"};
    auto ports = validPorts();
    EXPECT_STREQ(primitive.getName(), "Gdn2ChunkForwardBf16");
    EXPECT_TRUE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx936).isSuccess());
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    ports[1].type = FTRAIN_NUMERIC_TYPE_FP32;
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx936).isSuccess());
    ports = validPorts();
    for (auto& port : ports) port.dims[1] = 511;
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx936).isSuccess());
    ports = validPorts();
    ports[2].memory = reinterpret_cast<void*>(0x1002);
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx936).isSuccess());
}

TEST(Gdn2ChunkForward, WorkspaceIsCallerOwned) {
    const ftrain::Gdn2ChunkForward prototype;
    const auto problem = ftrain_test::problem<kKind>(validPorts());
    auto primitive = prototype.clone();
    static_cast<ftrain::Gdn2ChunkForward*>(primitive.get())->configure(problem);
    EXPECT_GT(primitive->getRequiredWorkspaceBytes(), 0U);
}

}  // namespace
