// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_PROJ_RMS_COMPUTE_H_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_PROJ_RMS_COMPUTE_H_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class ProjRmsComputeHFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &compute_h_eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &n_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &proj_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &r_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_res_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_post_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_pre_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &bias_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &alpha_res_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &alpha_post_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &alpha_pre_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &weight_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(
            ftrainPatternAddProjRmsComputeHFwd(pattern_, &proj_rms_compute_h_fwd_id_, x_id_, weight_id_, alpha_pre_id_,
                                               alpha_post_id_, alpha_res_id_, bias_id_, h_pre_id_, h_post_id_,
                                               h_res_id_, r_id_, proj_id_, n_id_, eps_id_, compute_h_eps_id_),
            FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView x, FTrainStorageView weight, FTrainStorageView alpha_pre,
                 FTrainStorageView alpha_post, FTrainStorageView alpha_res, FTrainStorageView bias,
                 FTrainStorageView h_pre, FTrainStorageView h_post, FTrainStorageView h_res, FTrainStorageView r,
                 FTrainStorageView proj, FTrainStorageView n, FTrainStorageView eps, FTrainStorageView compute_h_eps,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus x_status             = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus weight_status        = ftrainArgsSetTensor(args, weight_id_, weight);
        const FTrainStatus alpha_pre_status     = ftrainArgsSetTensor(args, alpha_pre_id_, alpha_pre);
        const FTrainStatus alpha_post_status    = ftrainArgsSetTensor(args, alpha_post_id_, alpha_post);
        const FTrainStatus alpha_res_status     = ftrainArgsSetTensor(args, alpha_res_id_, alpha_res);
        const FTrainStatus bias_status          = ftrainArgsSetTensor(args, bias_id_, bias);
        const FTrainStatus h_pre_status         = ftrainArgsSetTensor(args, h_pre_id_, h_pre);
        const FTrainStatus h_post_status        = ftrainArgsSetTensor(args, h_post_id_, h_post);
        const FTrainStatus h_res_status         = ftrainArgsSetTensor(args, h_res_id_, h_res);
        const FTrainStatus r_status             = ftrainArgsSetTensor(args, r_id_, r);
        const FTrainStatus proj_status          = ftrainArgsSetTensor(args, proj_id_, proj);
        const FTrainStatus n_status             = ftrainArgsSetTensor(args, n_id_, n);
        const FTrainStatus eps_status           = ftrainArgsSetTensor(args, eps_id_, eps);
        const FTrainStatus compute_h_eps_status = ftrainArgsSetTensor(args, compute_h_eps_id_, compute_h_eps);
        const FTrainStatus op_status = ftrainArgsSetProjRmsComputeHFwd(args, proj_rms_compute_h_fwd_id_, compute_type);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(weight_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(alpha_pre_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(alpha_post_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(alpha_res_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(bias_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_pre_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_post_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_res_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(r_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(proj_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(n_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(compute_h_eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return x_status == FTRAIN_STATUS_SUCCESS && weight_status == FTRAIN_STATUS_SUCCESS &&
               alpha_pre_status == FTRAIN_STATUS_SUCCESS && alpha_post_status == FTRAIN_STATUS_SUCCESS &&
               alpha_res_status == FTRAIN_STATUS_SUCCESS && bias_status == FTRAIN_STATUS_SUCCESS &&
               h_pre_status == FTRAIN_STATUS_SUCCESS && h_post_status == FTRAIN_STATUS_SUCCESS &&
               h_res_status == FTRAIN_STATUS_SUCCESS && r_status == FTRAIN_STATUS_SUCCESS &&
               proj_status == FTRAIN_STATUS_SUCCESS && n_status == FTRAIN_STATUS_SUCCESS &&
               eps_status == FTRAIN_STATUS_SUCCESS && compute_h_eps_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId x_id_{};
    FTrainTensorId weight_id_{};
    FTrainTensorId alpha_pre_id_{};
    FTrainTensorId alpha_post_id_{};
    FTrainTensorId alpha_res_id_{};
    FTrainTensorId bias_id_{};
    FTrainTensorId h_pre_id_{};
    FTrainTensorId h_post_id_{};
    FTrainTensorId h_res_id_{};
    FTrainTensorId r_id_{};
    FTrainTensorId proj_id_{};
    FTrainTensorId n_id_{};
    FTrainTensorId eps_id_{};
    FTrainTensorId compute_h_eps_id_{};
    FTrainProjRmsComputeHFwdOpId proj_rms_compute_h_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_PROJ_RMS_COMPUTE_H_FWD_STAGED_FIXTURE_HPP_
