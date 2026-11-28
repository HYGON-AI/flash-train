// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>

#include "flash_train/constraints.hpp"
#include "../family/gdn/problem_cases.hpp"

namespace {

template<ftrain::OperationKind Kind>
struct StagedCase {
    static constexpr std::size_t kPorts = Kind == ftrain::OperationKind::kGdnForward ? 8
                                          : Kind == ftrain::OperationKind::kGdnBackward ? 14
                                          : Kind == ftrain::OperationKind::kGdn2Forward ? 7 : 13;
    explicit StagedCase(std::vector<ftrain_test::Port> values) : specs(std::move(values)), ids(kPorts) {
        status = ftrainPatternCreate(&pattern);
        for (auto& id : ids) {
            if (status == FTRAIN_STATUS_SUCCESS) status = ftrainPatternAddTensor(pattern, &id);
        }
        if (status == FTRAIN_STATUS_SUCCESS) {
            if constexpr (Kind == ftrain::OperationKind::kGdnForward) {
                status = ftrainPatternAddGdnForward(pattern, &gdn_forward, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                   ids[5], ids[6], ids[7]);
            } else if constexpr (Kind == ftrain::OperationKind::kGdnBackward) {
                status = ftrainPatternAddGdnBackward(pattern, &gdn_backward, ids[0], ids[1], ids[2], ids[3],
                                                    ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10],
                                                    ids[11], ids[12], ids[13]);
            } else if constexpr (Kind == ftrain::OperationKind::kGdn2Forward) {
                status = ftrainPatternAddGdn2Forward(pattern, &gdn2_forward, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                    ids[5], ids[6]);
            } else {
                status = ftrainPatternAddGdn2Backward(pattern, &gdn2_backward, ids[0], ids[1], ids[2], ids[3],
                                                     ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10],
                                                     ids[11], ids[12]);
            }
        }
        if (status == FTRAIN_STATUS_SUCCESS) status = ftrainOpsCreate(&ops, pattern);
        if (status == FTRAIN_STATUS_SUCCESS) status = ftrainArgsCreate(&args, ops);
        for (std::size_t i = 0; i < kPorts && status == FTRAIN_STATUS_SUCCESS; ++i) {
            status = ftrainArgsSetTensor(args, ids[i], view(specs[i]));
        }
        if (status == FTRAIN_STATUS_SUCCESS) {
            if constexpr (Kind == ftrain::OperationKind::kGdnForward) {
                status = ftrainArgsSetGdnForward(args, gdn_forward, 0.5F);
            } else if constexpr (Kind == ftrain::OperationKind::kGdnBackward) {
                status = ftrainArgsSetGdnBackward(args, gdn_backward, 0.5F);
            } else if constexpr (Kind == ftrain::OperationKind::kGdn2Forward) {
                status = ftrainArgsSetGdn2Forward(args, gdn2_forward, 0.5F);
            } else {
                status = ftrainArgsSetGdn2Backward(args, gdn2_backward, 0.5F);
            }
        }
    }

    ~StagedCase() {
        if (args) ftrainArgsDestroy(args);
        if (ops) ftrainOpsDestroy(ops);
        if (pattern) ftrainPatternDestroy(pattern);
    }

    StagedCase(const StagedCase&) = delete;
    StagedCase& operator=(const StagedCase&) = delete;

    static FTrainStorageView view(const ftrain_test::Port& port) {
        return {port.memory, port.dims.data(), nullptr, static_cast<std::uint8_t>(port.dims.size()), port.type,
                FTRAIN_INDEX_TYPE_CONTINUOUS, port.host};
    }

    FTrainStatus planStatus() {
        FTrainPlan plan = nullptr;
        const FTrainStatus result = ftrainPlanCreate(&plan, args);
        if (plan) ftrainPlanDestroy(plan);
        return result;
    }

    std::vector<ftrain_test::Port> specs;
    std::vector<FTrainTensorId> ids;
    FTrainPattern pattern{nullptr};
    FTrainOps ops{nullptr};
    FTrainArgs args{nullptr};
    FTrainGdnForwardOpId gdn_forward{};
    FTrainGdnBackwardOpId gdn_backward{};
    FTrainGdn2ForwardOpId gdn2_forward{};
    FTrainGdn2BackwardOpId gdn2_backward{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

template<ftrain::OperationKind Kind>
FTrainStatus convenient(const std::vector<ftrain_test::Port>& specs) {
    std::array<FTrainStorageView, StagedCase<Kind>::kPorts> views{};
    for (std::size_t i = 0; i < views.size(); ++i) views[i] = StagedCase<Kind>::view(specs[i]);
    if constexpr (Kind == ftrain::OperationKind::kGdnForward) {
        return ftrainGdnForward(views[0], views[1], views[2], views[3], views[4], views[5], views[6], views[7],
                                0.5F, nullptr, 0, nullptr);
    } else if constexpr (Kind == ftrain::OperationKind::kGdnBackward) {
        return ftrainGdnBackward(views[0], views[1], views[2], views[3], views[4], views[5], views[6], views[7],
                                 views[8], views[9], views[10], views[11], views[12], views[13], 0.5F,
                                 nullptr, 0, nullptr);
    } else if constexpr (Kind == ftrain::OperationKind::kGdn2Forward) {
        return ftrainGdn2Forward(views[0], views[1], views[2], views[3], views[4], views[5], views[6], 0.5F,
                                 nullptr, 0, nullptr);
    } else {
        return ftrainGdn2Backward(views[0], views[1], views[2], views[3], views[4], views[5], views[6], views[7],
                                  views[8], views[9], views[10], views[11], views[12], 0.5F, nullptr, 0, nullptr);
    }
}

// Expected plan status for shapes that sit below the gfx936 chunk gates
// (the default problem_cases ports: S = 64, K = 8, V = 16, and their kin).
// gfx936 and unknown architectures keep rejecting them; the gfx938 routes
// pad any S >= 1 with K/V inside their caps, so they plan successfully.
FTrainStatus statusBelowGfx936ChunkGate() {
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    return arch == "gfx938" ? FTRAIN_STATUS_SUCCESS : FTRAIN_STATUS_UNSUPPORTED;
}

// The convenient API executes primitive 0 with the caller's workspace.  The
// gfx938 primitives require one, so with this test's null workspace the
// execution guard rejects the launch with INVALID_ARGUMENT while the plan
// itself succeeds (see statusBelowGfx936ChunkGate).
FTrainStatus convenientStatusBelowGfx936ChunkGate() {
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    return arch == "gfx938" ? FTRAIN_STATUS_INVALID_ARGUMENT : FTRAIN_STATUS_UNSUPPORTED;
}

#define FTRAIN_API_CASES(NAME, KIND)                                                                  \
    TEST(GdnApi, NAME##StagedUnsupported) {                                                           \
        StagedCase<KIND> test{ftrain_test::ports<KIND>()};                                             \
        ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);                                                \
        EXPECT_EQ(test.planStatus(), statusBelowGfx936ChunkGate());                                   \
    }                                                                                                  \
    TEST(GdnApi, NAME##ConvenientUnsupported) {                                                       \
        EXPECT_EQ(convenient<KIND>(ftrain_test::ports<KIND>()),                                       \
                  convenientStatusBelowGfx936ChunkGate());                                            \
    }                                                                                                  \
    TEST(GdnApi, NAME##InvalidShape) {                                                                \
        auto specs = ftrain_test::ports<KIND>(); specs[1].dims[3] = 9;                                \
        StagedCase<KIND> test{specs};                                                                  \
        ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);                                                \
        EXPECT_EQ(test.planStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);                                 \
        EXPECT_EQ(convenient<KIND>(specs), FTRAIN_STATUS_INVALID_ARGUMENT);                           \
    }

FTRAIN_API_CASES(GdnBackward, ftrain::OperationKind::kGdnBackward)
FTRAIN_API_CASES(Gdn2Forward, ftrain::OperationKind::kGdn2Forward)
FTRAIN_API_CASES(Gdn2Backward, ftrain::OperationKind::kGdn2Backward)

TEST(GdnApi, GdnForwardValidShapeHasCandidateOnSupportedArch) {
    StagedCase<ftrain::OperationKind::kGdnForward> test{
        ftrain_test::ports<ftrain::OperationKind::kGdnForward>()};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    EXPECT_EQ(test.planStatus(), (arch == "gfx936" || arch == "gfx938") ? FTRAIN_STATUS_SUCCESS
                                                                       : FTRAIN_STATUS_UNSUPPORTED);
}

TEST(GdnApi, GdnForwardRejectsNonChunkAlignedSequence) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (i != 5 && i != 7) specs[i].dims[1] = 65;
    }
    StagedCase<ftrain::OperationKind::kGdnForward> test{specs};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(test.planStatus(), statusBelowGfx936ChunkGate());
    // The convenient API really launches the primitive; the fused forward
    // needs no caller workspace, so on gfx938 it would run kernels on this
    // test's fake pointers.  Kernel-level coverage of the gfx938 padded
    // route lives in the numerics tests, which use real device buffers.
    if (ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch() != "gfx938") {
        EXPECT_EQ(convenient<ftrain::OperationKind::kGdnForward>(specs), FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(GdnApi, GdnForwardRejectsInvalidShape) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    specs[1].dims[3] = 9;
    StagedCase<ftrain::OperationKind::kGdnForward> test{specs};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(test.planStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(GdnApi, GdnFourForwardIsFirstAndNeedsCallerWorkspace) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (i == 5 || i == 7) continue;
        specs[i].dims[0] = 1;
        specs[i].dims[1] = 512;
        if (i <= 2 || i == 6) specs[i].dims.back() = 64;
    }
    specs[0].dims[2] = specs[1].dims[2] = 1;
    specs[2].dims[2] = specs[3].dims[2] = specs[4].dims[2] = specs[6].dims[2] = 1;
    StagedCase<ftrain::OperationKind::kGdnForward> test{specs};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    FTrainPlan plan = nullptr;
    const auto status = ftrainPlanCreate(&plan, test.args);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    if (arch == "gfx938") {
        // The gfx938 entry dispatches the four-kernel route internally: one
        // candidate, no caller workspace.
        EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(plan, nullptr);
        uint64_t count = 0, required = 0;
        const char* name = nullptr;
        EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(count, 1U);
        EXPECT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(name, nullptr);
        EXPECT_STREQ(name, "GdnFusedForwardBf16Gfx938");
        EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(required, 0U);
        EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        return;
    }
    if (arch != "gfx936") {
        EXPECT_EQ(status, FTRAIN_STATUS_UNSUPPORTED);
        return;
    }
    ASSERT_EQ(status, FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    uint64_t count = 0, required = 0, fused_required = 1;
    const char* name = nullptr;
    EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(name, nullptr);
    EXPECT_STREQ(name, "GdnFourForwardBf16");
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
    EXPECT_GT(required, 0U);
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 1, &fused_required), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(fused_required, 0U);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
}

TEST(GdnApi, Gdn2ForwardPlanReportsCallerWorkspace) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdn2Forward>();
    for (auto& port : specs) {
        port.dims[0] = 1;
        port.dims[1] = 512;
        port.dims.back() = 64;
    }
    specs[0].dims[2] = specs[1].dims[2] = 1;
    for (std::size_t i = 2; i < specs.size(); ++i) specs[i].dims[2] = 2;
    StagedCase<ftrain::OperationKind::kGdn2Forward> test{specs};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    FTrainPlan plan = nullptr;
    const auto status = ftrainPlanCreate(&plan, test.args);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    if (arch == "gfx938") {
        EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(plan, nullptr);
        uint64_t count = 0, required = 0;
        const char* name = nullptr;
        EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(count, 1U);
        EXPECT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(name, nullptr);
        EXPECT_STREQ(name, "Gdn2ChunkForwardBf16Gfx938");
        EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
        EXPECT_GT(required, 0U);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
                  FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        return;
    }
    if (arch != "gfx936") {
        EXPECT_EQ(status, FTRAIN_STATUS_UNSUPPORTED);
        return;
    }
    ASSERT_EQ(status, FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    uint64_t required = 0;
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
    EXPECT_GT(required, 0U);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
}

TEST(GdnApi, Gdn2BackwardPlanReportsCallerWorkspace) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdn2Backward>();
    for (auto& port : specs) {
        port.dims[0] = 1;
        port.dims[1] = 512;
        port.dims.back() = 64;
        port.dims[2] = 1;
    }
    StagedCase<ftrain::OperationKind::kGdn2Backward> test{specs};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    FTrainPlan plan = nullptr;
    const auto status = ftrainPlanCreate(&plan, test.args);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    if (arch == "gfx938") {
        EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(plan, nullptr);
        uint64_t count = 0, required = 0;
        const char* name = nullptr;
        EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(count, 1U);
        EXPECT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
        ASSERT_NE(name, nullptr);
        EXPECT_STREQ(name, "Gdn2ChunkBackwardBf16Gfx938");
        EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
        EXPECT_GT(required, 0U);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
                  FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        return;
    }
    if (arch != "gfx936") {
        EXPECT_EQ(status, FTRAIN_STATUS_UNSUPPORTED);
        return;
    }
    ASSERT_EQ(status, FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    uint64_t required = 0;
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
    EXPECT_GT(required, 0U);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
}

TEST(GdnApi, GdnBackwardStateAddsCallerWorkspace) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnBackward>();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (i == 6 || i == 7 || i == 13) continue;
        specs[i].dims[0] = 1;
        specs[i].dims[1] = 512;
        if (i <= 2 || i == 5 || (i >= 8 && i <= 10)) specs[i].dims.back() = 64;
        specs[i].dims[2] = 1;
    }
    auto bytes = [](const std::vector<ftrain_test::Port>& ports) {
        StagedCase<ftrain::OperationKind::kGdnBackward> test{ports};
        EXPECT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
        FTrainPlan plan = nullptr;
        EXPECT_EQ(ftrainPlanCreate(&plan, test.args), FTRAIN_STATUS_SUCCESS);
        if (!plan) return uint64_t{0};
        uint64_t required = 0;
        EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, required - 1, nullptr),
                  FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        return required;
    };
    {
        const std::string state_arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
        if (state_arch != "gfx936" && state_arch != "gfx938") GTEST_SKIP();
    }
    const auto without_state = bytes(specs);
    for (std::size_t i : {6U, 7U, 13U}) {
        specs[i].dims = {1, 1, 64, 64};
        specs[i].memory = reinterpret_cast<void*>(0x1000);
    }
    EXPECT_GT(bytes(specs), without_state);
}

TEST(GdnApi, OptionalStateZeroElementViews) {
    auto forward = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    forward[5].dims = {2, 0, 8, 16};
    forward[7].dims = {0, 4, 8, 16};
    StagedCase<ftrain::OperationKind::kGdnForward> fwd{forward};
    ASSERT_EQ(fwd.status, FTRAIN_STATUS_SUCCESS);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    EXPECT_EQ(fwd.planStatus(), (arch == "gfx936" || arch == "gfx938") ? FTRAIN_STATUS_SUCCESS
                                                                      : FTRAIN_STATUS_UNSUPPORTED);
    auto backward = ftrain_test::ports<ftrain::OperationKind::kGdnBackward>();
    StagedCase<ftrain::OperationKind::kGdnBackward> bwd{backward};
    ASSERT_EQ(bwd.status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(bwd.planStatus(), statusBelowGfx936ChunkGate());
}

TEST(GdnApi, InvalidScaleIsRejectedByArgsSetter) {
    StagedCase<ftrain::OperationKind::kGdnForward> test{
        ftrain_test::ports<ftrain::OperationKind::kGdnForward>()};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGdnForward(test.args, test.gdn_forward, 0.0F), FTRAIN_STATUS_INVALID_ARGUMENT);
}

struct MatrixShape {
    const char* name;
    std::int64_t b, s, h, hv, k, v;
};

// Frozen from the two platform shape_matrix.json files for plan-only coverage.
constexpr std::array<MatrixShape, 31> kGdnMatrix{{
    {"small", 1, 64, 4, 4, 32, 32}, {"small2", 1, 128, 8, 8, 64, 64},
    {"tiny2", 2, 64, 4, 4, 64, 64}, {"tail", 1, 65, 4, 8, 37, 33},
    {"medium", 1, 256, 16, 32, 64, 64}, {"medium2", 1, 512, 16, 16, 128, 128},
    {"baseline", 1, 1024, 32, 32, 128, 128}, {"v256", 1, 1024, 16, 16, 128, 256},
    {"k64v256", 1, 1024, 16, 16, 64, 256}, {"gqa2v256", 1, 1024, 8, 16, 128, 256},
    {"narrowK", 1, 2048, 32, 32, 64, 128}, {"narrowV", 1, 2048, 32, 32, 128, 64},
    {"gqa4", 1, 2048, 8, 32, 128, 128}, {"llm2k", 1, 2048, 32, 32, 128, 128},
    {"wide", 2, 2048, 16, 32, 128, 128}, {"bigbatch", 8, 512, 32, 32, 128, 128},
    {"batch4", 4, 1024, 32, 32, 128, 128}, {"large", 1, 4096, 32, 32, 128, 128},
    {"llm4k", 1, 4096, 32, 32, 128, 128}, {"xlong", 1, 8192, 32, 32, 128, 128},
    {"llm8k", 1, 8192, 32, 32, 128, 128}, {"llm16k", 1, 16384, 32, 32, 128, 128},
    {"gqa8k", 1, 8192, 32, 8, 128, 128}, {"gqa32k", 1, 32768, 32, 8, 128, 128},
    {"llm32k_k64", 1, 32768, 32, 32, 64, 128}, {"llm32k", 1, 32768, 32, 32, 128, 128},
    {"llm8k_b8", 8, 8192, 32, 32, 128, 128}, {"llm64k", 1, 65536, 32, 32, 128, 128},
    {"llm8k_b16", 16, 8192, 32, 32, 128, 128},
    {"llm32k_b4", 4, 32768, 32, 32, 128, 128},
    {"llm128k", 1, 131072, 32, 32, 128, 128},
}};

constexpr std::array<MatrixShape, 34> kGdn2Matrix{{
    {"small", 1, 64, 4, 4, 32, 32}, {"small2", 1, 128, 8, 8, 64, 64},
    {"tiny2", 2, 64, 4, 4, 64, 64}, {"tail", 1, 65, 4, 8, 37, 33},
    {"medium", 1, 256, 16, 32, 64, 64}, {"medium2", 1, 512, 16, 16, 128, 128},
    {"baseline", 1, 1024, 32, 32, 128, 128}, {"v256", 1, 1024, 16, 16, 128, 256},
    {"narrowK", 1, 2048, 32, 32, 64, 128}, {"narrowV", 1, 2048, 32, 32, 128, 64},
    {"gqa4", 1, 2048, 8, 32, 128, 128}, {"llm2k", 1, 2048, 32, 32, 128, 128},
    {"wide", 2, 2048, 16, 32, 128, 128}, {"bigbatch", 8, 512, 32, 32, 128, 128},
    {"batch4", 4, 1024, 32, 32, 128, 128}, {"large", 1, 4096, 32, 32, 128, 128},
    {"llm4k", 1, 4096, 32, 32, 128, 128}, {"xlong", 1, 8192, 32, 32, 128, 128},
    {"llm8k", 1, 8192, 32, 32, 128, 128}, {"llm16k", 1, 16384, 32, 32, 128, 128},
    {"gqa8k", 1, 8192, 32, 8, 128, 128}, {"gqa32k", 1, 32768, 32, 8, 128, 128},
    {"llm32k_k64", 1, 32768, 32, 32, 64, 128}, {"llm32k", 1, 32768, 32, 32, 128, 128},
    {"llm8k_b8", 8, 8192, 32, 32, 128, 128}, {"llm64k", 1, 65536, 32, 32, 128, 128},
    {"llm8k_b16", 16, 8192, 32, 32, 128, 128},
    {"llm32k_b4", 4, 32768, 32, 32, 128, 128},
    {"llm128k", 1, 131072, 32, 32, 128, 128},
    {"gqa", 2, 256, 4, 16, 64, 64}, {"h16_s128", 1, 128, 16, 16, 128, 128},
    {"h16_s512", 1, 512, 16, 16, 128, 128},
    {"h16_s2048", 1, 2048, 16, 16, 128, 128},
    {"h16_s4096", 1, 4096, 16, 16, 128, 128},
}};

template<ftrain::OperationKind Kind>
void checkMatrixPlan(const MatrixShape& shape) {
    auto specs = ftrain_test::ports<Kind>();
    const ftrain_test::Dims q{shape.b, shape.s, shape.h, shape.k};
    const ftrain_test::Dims v{shape.b, shape.s, shape.hv, shape.v};
    const ftrain_test::Dims gate{shape.b, shape.s, shape.hv};
    const ftrain_test::Dims gate_k{shape.b, shape.s, shape.hv, shape.k};
    const ftrain_test::Dims empty{0, shape.hv, shape.k, shape.v};
    specs[0].dims = specs[1].dims = q;
    specs[2].dims = v;
    if constexpr (Kind == ftrain::OperationKind::kGdnForward) {
        specs[3].dims = specs[4].dims = gate;
        specs[5].dims = specs[7].dims = empty;
        specs[6].dims = v;
    } else if constexpr (Kind == ftrain::OperationKind::kGdnBackward) {
        specs[3].dims = specs[4].dims = specs[11].dims = specs[12].dims = gate;
        specs[5].dims = specs[10].dims = v;
        specs[6].dims = specs[7].dims = specs[13].dims = empty;
        specs[8].dims = specs[9].dims = q;
    } else if constexpr (Kind == ftrain::OperationKind::kGdn2Forward) {
        specs[3].dims = specs[4].dims = gate_k;
        specs[5].dims = specs[6].dims = v;
    } else {
        specs[3].dims = specs[4].dims = specs[10].dims = specs[11].dims = gate_k;
        specs[5].dims = specs[6].dims = specs[9].dims = specs[12].dims = v;
        specs[7].dims = specs[8].dims = q;
    }
    const bool chunk = shape.s >= 512 && shape.s % 64 == 0 &&
                       (shape.k == 64 || shape.k == 128) &&
                       (shape.v == 64 || shape.v == 128);
    const bool supported = Kind == ftrain::OperationKind::kGdnForward
                               ? shape.s % 32 == 0 && shape.k <= 128 && shape.v % 16 == 0
                               : chunk;
    // Frozen from the archive 05/13 entry gates: on gfx938 any S >= 1 is
    // covered through the padded route, K stays within [1, 128] and V within
    // [1, 256]; the GDN backward additionally pins V = 256 to the native
    // path (S % 64 == 0, K in {64, 128}).
    const bool gfx938_supported =
        shape.s >= 1 && shape.k >= 1 && shape.k <= 128 &&
        (Kind == ftrain::OperationKind::kGdnBackward
             ? shape.v >= 1 && (shape.v <= 128 ||
                                (shape.v == 256 && shape.s % 64 == 0 && (shape.k == 64 || shape.k == 128)))
             : shape.v >= 1 && shape.v <= 256);
    const std::string arch = ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch();
    StagedCase<Kind> test{std::move(specs)};
    ASSERT_EQ(test.status, FTRAIN_STATUS_SUCCESS);
    FTrainStatus expected = shape.hv % shape.h != 0 ? FTRAIN_STATUS_INVALID_ARGUMENT
                                                    : FTRAIN_STATUS_UNSUPPORTED;
    if (shape.hv % shape.h == 0) {
        if (arch == "gfx936") {
            expected = supported ? FTRAIN_STATUS_SUCCESS : FTRAIN_STATUS_UNSUPPORTED;
        } else if (arch == "gfx938") {
            expected = gfx938_supported ? FTRAIN_STATUS_SUCCESS : FTRAIN_STATUS_UNSUPPORTED;
        }
    }
    EXPECT_EQ(test.planStatus(), expected);
}

TEST(GdnApi, PlatformShapeMatrixPlanCoverage) {
    for (const auto& shape : kGdnMatrix) {
        SCOPED_TRACE(shape.name);
        checkMatrixPlan<ftrain::OperationKind::kGdnForward>(shape);
        checkMatrixPlan<ftrain::OperationKind::kGdnBackward>(shape);
    }
    for (const auto& shape : kGdn2Matrix) {
        SCOPED_TRACE(shape.name);
        checkMatrixPlan<ftrain::OperationKind::kGdn2Forward>(shape);
        checkMatrixPlan<ftrain::OperationKind::kGdn2Backward>(shape);
    }
}

}  // namespace
