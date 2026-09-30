// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef FTRAIN_TESTS_GEMM_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_GEMM_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

// Gemm fixture on top of StagedApiTest: pattern assembly plus role ids and
// the six-port args binding. Behavior tests and numerics tests share it.
class GemmStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);

        // Deliberately differs from the built-in PatternBuilder's operand-add order.
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &d_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &beta_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &alpha_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &c_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &b_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &a_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddGemm(pattern_, &gemm_id_, a_id_, b_id_, c_id_, d_id_, alpha_id_, beta_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);

        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                 FTrainStorageView alpha, FTrainStorageView beta,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus a_status     = ftrainArgsSetTensor(args, a_id_, a);
        const FTrainStatus b_status     = ftrainArgsSetTensor(args, b_id_, b);
        const FTrainStatus c_status     = ftrainArgsSetTensor(args, c_id_, c);
        const FTrainStatus d_status     = ftrainArgsSetTensor(args, d_id_, d);
        const FTrainStatus alpha_status = ftrainArgsSetTensor(args, alpha_id_, alpha);
        const FTrainStatus beta_status  = ftrainArgsSetTensor(args, beta_id_, beta);
        const FTrainStatus op_status    = ftrainArgsSetGemm(args, gemm_id_, compute_type);
        EXPECT_EQ(a_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(b_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(c_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(d_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(alpha_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(beta_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return a_status == FTRAIN_STATUS_SUCCESS && b_status == FTRAIN_STATUS_SUCCESS &&
               c_status == FTRAIN_STATUS_SUCCESS && d_status == FTRAIN_STATUS_SUCCESS &&
               alpha_status == FTRAIN_STATUS_SUCCESS && beta_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId a_id_{};
    FTrainTensorId b_id_{};
    FTrainTensorId c_id_{};
    FTrainTensorId d_id_{};
    FTrainTensorId alpha_id_{};
    FTrainTensorId beta_id_{};
    FTrainGemmOpId gemm_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_GEMM_STAGED_FIXTURE_HPP_
