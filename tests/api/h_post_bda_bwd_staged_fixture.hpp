// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_H_POST_BDA_BWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_H_POST_BDA_BWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class HPostBdaBwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_bias_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_h_post_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_h_res_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_residual_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_post_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_res_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &bias_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &residual_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_output_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddHPostBdaBwd(pattern_, &h_post_bda_bwd_id_, grad_output_id_, residual_id_, x_id_,
                                              bias_id_, h_res_id_, h_post_id_, grad_residual_id_, grad_x_id_,
                                              grad_h_res_id_, grad_h_post_id_, grad_bias_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView grad_output, FTrainStorageView residual, FTrainStorageView x,
                 FTrainStorageView bias, FTrainStorageView h_res, FTrainStorageView h_post,
                 FTrainStorageView grad_residual, FTrainStorageView grad_x, FTrainStorageView grad_h_res,
                 FTrainStorageView grad_h_post, FTrainStorageView grad_bias,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus grad_output_status   = ftrainArgsSetTensor(args, grad_output_id_, grad_output);
        const FTrainStatus residual_status      = ftrainArgsSetTensor(args, residual_id_, residual);
        const FTrainStatus x_status             = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus bias_status          = ftrainArgsSetTensor(args, bias_id_, bias);
        const FTrainStatus h_res_status         = ftrainArgsSetTensor(args, h_res_id_, h_res);
        const FTrainStatus h_post_status        = ftrainArgsSetTensor(args, h_post_id_, h_post);
        const FTrainStatus grad_residual_status = ftrainArgsSetTensor(args, grad_residual_id_, grad_residual);
        const FTrainStatus grad_x_status        = ftrainArgsSetTensor(args, grad_x_id_, grad_x);
        const FTrainStatus grad_h_res_status    = ftrainArgsSetTensor(args, grad_h_res_id_, grad_h_res);
        const FTrainStatus grad_h_post_status   = ftrainArgsSetTensor(args, grad_h_post_id_, grad_h_post);
        const FTrainStatus grad_bias_status     = ftrainArgsSetTensor(args, grad_bias_id_, grad_bias);
        const FTrainStatus op_status            = ftrainArgsSetHPostBdaBwd(args, h_post_bda_bwd_id_, compute_type);
        EXPECT_EQ(grad_output_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(residual_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(bias_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_res_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_post_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_residual_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_h_res_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_h_post_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_bias_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return grad_output_status == FTRAIN_STATUS_SUCCESS && residual_status == FTRAIN_STATUS_SUCCESS &&
               x_status == FTRAIN_STATUS_SUCCESS && bias_status == FTRAIN_STATUS_SUCCESS &&
               h_res_status == FTRAIN_STATUS_SUCCESS && h_post_status == FTRAIN_STATUS_SUCCESS &&
               grad_residual_status == FTRAIN_STATUS_SUCCESS && grad_x_status == FTRAIN_STATUS_SUCCESS &&
               grad_h_res_status == FTRAIN_STATUS_SUCCESS && grad_h_post_status == FTRAIN_STATUS_SUCCESS &&
               grad_bias_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId grad_output_id_{};
    FTrainTensorId residual_id_{};
    FTrainTensorId x_id_{};
    FTrainTensorId bias_id_{};
    FTrainTensorId h_res_id_{};
    FTrainTensorId h_post_id_{};
    FTrainTensorId grad_residual_id_{};
    FTrainTensorId grad_x_id_{};
    FTrainTensorId grad_h_res_id_{};
    FTrainTensorId grad_h_post_id_{};
    FTrainTensorId grad_bias_id_{};
    FTrainHPostBdaBwdOpId h_post_bda_bwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_H_POST_BDA_BWD_STAGED_FIXTURE_HPP_
