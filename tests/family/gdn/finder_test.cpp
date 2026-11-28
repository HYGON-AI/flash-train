// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "flash_train/family/gdn/finder.hpp"
#include "problem_cases.hpp"

namespace {

TEST(GdnFinder, ReturnsRegisteredForwardCandidatesInPreferenceOrder) {
    const auto problem = ftrain_test::problem<ftrain::OperationKind::kGdnForward>(
        ftrain_test::ports<ftrain::OperationKind::kGdnForward>());
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnForward>::findCandidates(
            problem, ftrain::Constraints{0, "gfx936"})),
        (std::vector<std::string>{"GdnFusedForwardBf16"}));
}

TEST(GdnFinder, PrefersFourKernelForItsShape) {
    auto ports = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    for (auto& port : ports) {
        if (port.dims.size() >= 2) port.dims[1] = 512;
    }
    ports[0].dims[2] = ports[1].dims[2] = 1;
    ports[0].dims[3] = ports[1].dims[3] = 64;
    ports[2].dims[2] = ports[6].dims[2] = 1;
    ports[2].dims[3] = ports[6].dims[3] = 64;
    ports[3].dims[2] = ports[4].dims[2] = 1;
    ports[5].dims = {0, 4, 8, 16};
    ports[7].dims = {0, 4, 8, 16};
    const auto problem = ftrain_test::problem<ftrain::OperationKind::kGdnForward>(ports);
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnForward>::findCandidates(
            problem, ftrain::Constraints{0, "gfx936"})),
        (std::vector<std::string>{"GdnFourForwardBf16", "GdnFusedForwardBf16"}));
}

TEST(GdnFinder, ReturnsNamesForBackwardAndGdn2) {
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnBackward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdnBackward>(
                ftrain_test::ports<ftrain::OperationKind::kGdnBackward>()),
            ftrain::Constraints{0, "gfx936"})),
        (std::vector<std::string>{"GdnChunkBackwardBf16"}));
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdn2Forward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdn2Forward>(
                ftrain_test::ports<ftrain::OperationKind::kGdn2Forward>()),
            ftrain::Constraints{0, "gfx936"})),
        (std::vector<std::string>{"Gdn2ChunkForwardBf16"}));
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdn2Backward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdn2Backward>(
                ftrain_test::ports<ftrain::OperationKind::kGdn2Backward>()),
            ftrain::Constraints{0, "gfx936"})),
        (std::vector<std::string>{"Gdn2ChunkBackwardBf16"}));
}

#if FTRAIN_HAS_GFX938
// The gfx938 branch names the single candidate per kind that the archive
// entry provides; the forward four-kernel route is dispatched inside the
// gfx938 entry, so no separate candidate exists.  Only compiled when the
// gfx938 records are registered: in a gfx936-only build the branch returns
// an empty list and the guarded fallback test below covers that contract.
TEST(GdnFinder, Gfx938CandidatesPerKind) {
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnForward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdnForward>(
                ftrain_test::ports<ftrain::OperationKind::kGdnForward>()),
            ftrain::Constraints{0, "gfx938"})),
        (std::vector<std::string>{"GdnFusedForwardBf16Gfx938"}));
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnBackward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdnBackward>(
                ftrain_test::ports<ftrain::OperationKind::kGdnBackward>()),
            ftrain::Constraints{0, "gfx938"})),
        (std::vector<std::string>{"GdnChunkBackwardBf16Gfx938"}));
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdn2Forward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdn2Forward>(
                ftrain_test::ports<ftrain::OperationKind::kGdn2Forward>()),
            ftrain::Constraints{0, "gfx938"})),
        (std::vector<std::string>{"Gdn2ChunkForwardBf16Gfx938"}));
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdn2Backward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdn2Backward>(
                ftrain_test::ports<ftrain::OperationKind::kGdn2Backward>()),
            ftrain::Constraints{0, "gfx938"})),
        (std::vector<std::string>{"Gdn2ChunkBackwardBf16Gfx938"}));
}
#endif

#if !FTRAIN_HAS_GFX938
TEST(GdnFinder, Gfx938ArchWithoutGfx938BuildFallsBackToRegistrationOrder) {
    // The gfx938 branch contributes nothing, so the documented
    // registration-order fallback runs and every record's rejection is named.
    EXPECT_EQ(
        (ftrain::GdnRegistrationOrderFinder<ftrain::OperationKind::kGdnForward>::findCandidates(
            ftrain_test::problem<ftrain::OperationKind::kGdnForward>(
                ftrain_test::ports<ftrain::OperationKind::kGdnForward>()),
            ftrain::Constraints{0, "gfx938"})),
        std::vector<std::string>{});
}
#endif

}  // namespace
