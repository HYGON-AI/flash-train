// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "../gdn/problem_cases.hpp"

FTRAIN_GDN_PROBLEM_CASES(Gdn2ForwardProblem, ftrain::OperationKind::kGdn2Forward)
FTRAIN_GDN_PROBLEM_CASES(Gdn2BackwardProblem, ftrain::OperationKind::kGdn2Backward)

TEST(Gdn2ForwardProblem, RejectOutputTypeMismatch) {
    auto specs = ftrain_test::ports<ftrain::OperationKind::kGdn2Forward>();
    specs[6].type = FTRAIN_NUMERIC_TYPE_FP32;
    ftrain_test::expectStatus<ftrain::OperationKind::kGdn2Forward>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);
}

#define EXPECT_PORT(Problem, Name, Port)                                    \
    EXPECT_EQ((Problem).get##Name##Address(), (Problem).getPortAddress(Port)); \
    EXPECT_EQ((Problem).get##Name##NumericType(), (Problem).getPortNumericType(Port)); \
    EXPECT_EQ((Problem).get##Name##IndexType(), (Problem).getPortIndexType(Port)); \
    EXPECT_EQ((Problem).get##Name##Strides(), (Problem).getPortStrides(Port)); \
    EXPECT_EQ((Problem).get##Name##Alignment(), (Problem).getPortAlignment(Port)); \
    EXPECT_EQ((Problem).is##Name##Continuous(), (Problem).isPortContinuous(Port))

TEST(Gdn2ForwardProblem, SemanticPorts) {
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdn2Forward>(
        ftrain_test::ports<ftrain::OperationKind::kGdn2Forward>());
    EXPECT_PORT(p, Q, 0); EXPECT_PORT(p, K, 1); EXPECT_PORT(p, V, 2);
    EXPECT_PORT(p, G, 3); EXPECT_PORT(p, BGate, 4); EXPECT_PORT(p, W, 5);
    EXPECT_PORT(p, Out, 6);
}

TEST(Gdn2BackwardProblem, SemanticPorts) {
    auto p = ftrain_test::problem<ftrain::OperationKind::kGdn2Backward>(
        ftrain_test::ports<ftrain::OperationKind::kGdn2Backward>());
    EXPECT_PORT(p, Q, 0); EXPECT_PORT(p, K, 1); EXPECT_PORT(p, V, 2);
    EXPECT_PORT(p, G, 3); EXPECT_PORT(p, BGate, 4); EXPECT_PORT(p, W, 5);
    EXPECT_PORT(p, Do, 6); EXPECT_PORT(p, DQ, 7); EXPECT_PORT(p, DK, 8);
    EXPECT_PORT(p, DV, 9); EXPECT_PORT(p, DG, 10);
    EXPECT_PORT(p, DBGate, 11); EXPECT_PORT(p, DW, 12);
}

#undef EXPECT_PORT
