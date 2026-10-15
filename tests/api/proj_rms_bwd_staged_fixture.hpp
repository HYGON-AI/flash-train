// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_PROJ_RMS_BWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_PROJ_RMS_BWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class ProjRmsBwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_weight_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &r_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &weight_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_r_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_proj_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddProjRmsBwd(pattern_, &proj_rms_bwd_id_, grad_proj_id_, grad_r_id_, x_id_, weight_id_,
                                             r_id_, grad_x_id_, grad_weight_id_, eps_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView grad_proj, FTrainStorageView grad_r, FTrainStorageView x,
                 FTrainStorageView weight, FTrainStorageView r, FTrainStorageView grad_x, FTrainStorageView grad_weight,
                 FTrainStorageView eps, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus grad_proj_status   = ftrainArgsSetTensor(args, grad_proj_id_, grad_proj);
        const FTrainStatus grad_r_status      = ftrainArgsSetTensor(args, grad_r_id_, grad_r);
        const FTrainStatus x_status           = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus weight_status      = ftrainArgsSetTensor(args, weight_id_, weight);
        const FTrainStatus r_status           = ftrainArgsSetTensor(args, r_id_, r);
        const FTrainStatus grad_x_status      = ftrainArgsSetTensor(args, grad_x_id_, grad_x);
        const FTrainStatus grad_weight_status = ftrainArgsSetTensor(args, grad_weight_id_, grad_weight);
        const FTrainStatus eps_status         = ftrainArgsSetTensor(args, eps_id_, eps);
        const FTrainStatus op_status          = ftrainArgsSetProjRmsBwd(args, proj_rms_bwd_id_, compute_type);
        EXPECT_EQ(grad_proj_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_r_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(weight_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(r_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_weight_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return grad_proj_status == FTRAIN_STATUS_SUCCESS && grad_r_status == FTRAIN_STATUS_SUCCESS &&
               x_status == FTRAIN_STATUS_SUCCESS && weight_status == FTRAIN_STATUS_SUCCESS &&
               r_status == FTRAIN_STATUS_SUCCESS && grad_x_status == FTRAIN_STATUS_SUCCESS &&
               grad_weight_status == FTRAIN_STATUS_SUCCESS && eps_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId grad_proj_id_{};
    FTrainTensorId grad_r_id_{};
    FTrainTensorId x_id_{};
    FTrainTensorId weight_id_{};
    FTrainTensorId r_id_{};
    FTrainTensorId grad_x_id_{};
    FTrainTensorId grad_weight_id_{};
    FTrainTensorId eps_id_{};
    FTrainProjRmsBwdOpId proj_rms_bwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_PROJ_RMS_BWD_STAGED_FIXTURE_HPP_
