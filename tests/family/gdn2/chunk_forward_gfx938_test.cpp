// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "flash_train/primitive/hygon/gdn2/chunk_forward_gfx938.hpp"
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

TEST(Gdn2ChunkForwardGfx938, ShapeDtypeAndArchGate) {
    const ftrain::Gdn2ChunkForwardGfx938 primitive;
    const ftrain::Constraints gfx936{0, "gfx936"}, gfx938{0, "gfx938"};
    auto ports = validPorts();
    EXPECT_STREQ(primitive.getName(), "Gdn2ChunkForwardBf16Gfx938");
    EXPECT_TRUE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx936).isSuccess());
    ports[1].type = FTRAIN_NUMERIC_TYPE_FP32;
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    // Unlike the gfx936 candidate, any S >= 1 runs through the padded route.
    ports = validPorts();
    for (auto& port : ports) port.dims[1] = 100;
    EXPECT_TRUE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    // K > 128 stays unsupported.
    ports = validPorts();
    for (auto& port : ports) {
        if (port.dims.back() == 64) port.dims.back() = 192;
    }
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
    ports = validPorts();
    ports[2].memory = reinterpret_cast<void*>(0x1002);
    EXPECT_FALSE(primitive.isApplicable(ftrain_test::problem<kKind>(ports), gfx938).isSuccess());
}

TEST(Gdn2ChunkForwardGfx938, WorkspaceIsCallerOwned) {
    const ftrain::Gdn2ChunkForwardGfx938 prototype;
    const auto problem = ftrain_test::problem<kKind>(validPorts());
    auto primitive = prototype.clone();
    static_cast<ftrain::Gdn2ChunkForwardGfx938*>(primitive.get())->configure(problem);
    EXPECT_GT(primitive->getRequiredWorkspaceBytes(), 0U);
}

}  // namespace
