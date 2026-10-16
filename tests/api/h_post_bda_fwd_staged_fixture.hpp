// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_H_POST_BDA_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_H_POST_BDA_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class HPostBdaFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &output_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &bias_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_post_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &residual_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_res_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddHPostBdaFwd(pattern_, &h_post_bda_fwd_id_, h_res_id_, residual_id_, h_post_id_, x_id_,
                                              bias_id_, output_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView h_res, FTrainStorageView residual, FTrainStorageView h_post,
                 FTrainStorageView x, FTrainStorageView bias, FTrainStorageView output,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus h_res_status    = ftrainArgsSetTensor(args, h_res_id_, h_res);
        const FTrainStatus residual_status = ftrainArgsSetTensor(args, residual_id_, residual);
        const FTrainStatus h_post_status   = ftrainArgsSetTensor(args, h_post_id_, h_post);
        const FTrainStatus x_status        = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus bias_status     = ftrainArgsSetTensor(args, bias_id_, bias);
        const FTrainStatus output_status   = ftrainArgsSetTensor(args, output_id_, output);
        const FTrainStatus op_status       = ftrainArgsSetHPostBdaFwd(args, h_post_bda_fwd_id_, compute_type);
        EXPECT_EQ(h_res_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(residual_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_post_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(bias_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(output_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return h_res_status == FTRAIN_STATUS_SUCCESS && residual_status == FTRAIN_STATUS_SUCCESS &&
               h_post_status == FTRAIN_STATUS_SUCCESS && x_status == FTRAIN_STATUS_SUCCESS &&
               bias_status == FTRAIN_STATUS_SUCCESS && output_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId h_res_id_{};
    FTrainTensorId residual_id_{};
    FTrainTensorId h_post_id_{};
    FTrainTensorId x_id_{};
    FTrainTensorId bias_id_{};
    FTrainTensorId output_id_{};
    FTrainHPostBdaFwdOpId h_post_bda_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_H_POST_BDA_FWD_STAGED_FIXTURE_HPP_
