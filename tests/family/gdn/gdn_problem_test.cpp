// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "problem_cases.hpp"

FTRAIN_GDN_PROBLEM_CASES(GdnForwardProblem, ftrain::OperationKind::kGdnForward)
FTRAIN_GDN_PROBLEM_CASES(GdnBackwardProblem, ftrain::OperationKind::kGdnBackward)

TEST(GdnForwardProblem, OptionalStatePresenceAndValidation) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    EXPECT_FALSE(ftrain_test::problem<ftrain::OperationKind::kGdnForward>(specs).hasH0());
    specs[5].dims = {2, 4, 8, 16};
    specs[7].dims = {2, 4, 8, 16};
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdnForward>(specs);
    EXPECT_TRUE(p.hasH0());
    EXPECT_TRUE(p.hasHt());
    EXPECT_NE(p.getProblemKey(),
              ftrain_test::problem<ftrain::OperationKind::kGdnForward>(
                  ftrain_test::ports<ftrain::OperationKind::kGdnForward>()).getProblemKey());
    specs[5].type = FTRAIN_NUMERIC_TYPE_BF16;
    ftrain_test::expectStatus<ftrain::OperationKind::kGdnForward>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);
    specs[5].type = FTRAIN_NUMERIC_TYPE_FP32;
    specs[7].dims[3] = 17;
    ftrain_test::expectStatus<ftrain::OperationKind::kGdnForward>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(GdnForwardProblem, RejectOutputTypeMismatch) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnForward>();
    specs[6].type = FTRAIN_NUMERIC_TYPE_FP32;
    ftrain_test::expectStatus<ftrain::OperationKind::kGdnForward>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(GdnBackwardProblem, OptionalStatePresenceAndValidation) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdnBackward>();
    auto absent = ftrain_test::problem<ftrain::OperationKind::kGdnBackward>(specs);
    EXPECT_FALSE(absent.hasH0());
    EXPECT_FALSE(absent.hasDht());
    EXPECT_FALSE(absent.hasDh0());
    specs[6].dims = {2, 4, 8, 16};
    specs[7].dims = {2, 4, 8, 16};
    specs[13].dims = {2, 4, 8, 16};
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdnBackward>(specs);
    EXPECT_TRUE(p.hasH0());
    EXPECT_TRUE(p.hasDht());
    EXPECT_TRUE(p.hasDh0());
    EXPECT_NE(p.getProblemKey(), absent.getProblemKey());
    specs[13].type = FTRAIN_NUMERIC_TYPE_BF16;
    ftrain_test::expectStatus<ftrain::OperationKind::kGdnBackward>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);
}

#define EXPECT_PORT(Problem, Name, Port)                                    \
    EXPECT_EQ((Problem).get##Name##Address(), (Problem).getPortAddress(Port)); \
    EXPECT_EQ((Problem).get##Name##NumericType(), (Problem).getPortNumericType(Port)); \
    EXPECT_EQ((Problem).get##Name##IndexType(), (Problem).getPortIndexType(Port)); \
    EXPECT_EQ((Problem).get##Name##Strides(), (Problem).getPortStrides(Port)); \
    EXPECT_EQ((Problem).get##Name##Alignment(), (Problem).getPortAlignment(Port)); \
    EXPECT_EQ((Problem).is##Name##Continuous(), (Problem).isPortContinuous(Port))

TEST(GdnForwardProblem, SemanticPorts) {
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdnForward>(
        ftrain_test::ports<ftrain::OperationKind::kGdnForward>());
    EXPECT_PORT(p, Q, 0); EXPECT_PORT(p, K, 1); EXPECT_PORT(p, V, 2);
    EXPECT_PORT(p, G, 3); EXPECT_PORT(p, Beta, 4); EXPECT_PORT(p, H0, 5);
    EXPECT_PORT(p, Out, 6); EXPECT_PORT(p, Ht, 7);
}

TEST(GdnBackwardProblem, SemanticPorts) {
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdnBackward>(
        ftrain_test::ports<ftrain::OperationKind::kGdnBackward>());
    EXPECT_PORT(p, Q, 0); EXPECT_PORT(p, K, 1); EXPECT_PORT(p, V, 2);
    EXPECT_PORT(p, G, 3); EXPECT_PORT(p, Beta, 4); EXPECT_PORT(p, Do, 5);
    EXPECT_PORT(p, H0, 6); EXPECT_PORT(p, Dht, 7); EXPECT_PORT(p, DQ, 8);
    EXPECT_PORT(p, DK, 9); EXPECT_PORT(p, DV, 10); EXPECT_PORT(p, DG, 11);
    EXPECT_PORT(p, DBeta, 12); EXPECT_PORT(p, Dh0, 13);
}

#undef EXPECT_PORT
