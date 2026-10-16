// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_PROJ_RMS_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_PROJ_RMS_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class ProjRmsFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &r_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &proj_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &weight_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddProjRmsFwd(pattern_, &proj_rms_fwd_id_, x_id_, weight_id_, proj_id_, r_id_, eps_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView x, FTrainStorageView weight, FTrainStorageView proj,
                 FTrainStorageView r, FTrainStorageView eps,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus x_status      = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus weight_status = ftrainArgsSetTensor(args, weight_id_, weight);
        const FTrainStatus proj_status   = ftrainArgsSetTensor(args, proj_id_, proj);
        const FTrainStatus r_status      = ftrainArgsSetTensor(args, r_id_, r);
        const FTrainStatus eps_status    = ftrainArgsSetTensor(args, eps_id_, eps);
        const FTrainStatus op_status     = ftrainArgsSetProjRmsFwd(args, proj_rms_fwd_id_, compute_type);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(weight_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(proj_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(r_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return x_status == FTRAIN_STATUS_SUCCESS && weight_status == FTRAIN_STATUS_SUCCESS &&
               proj_status == FTRAIN_STATUS_SUCCESS && r_status == FTRAIN_STATUS_SUCCESS &&
               eps_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId x_id_{};
    FTrainTensorId weight_id_{};
    FTrainTensorId proj_id_{};
    FTrainTensorId r_id_{};
    FTrainTensorId eps_id_{};
    FTrainProjRmsFwdOpId proj_rms_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_PROJ_RMS_FWD_STAGED_FIXTURE_HPP_
