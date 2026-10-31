// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SINKHORN_BWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SINKHORN_BWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SinkhornBwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &num_iterations_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_logits_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &m_init_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_output_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddSinkhornBwd(pattern_, &sinkhorn_bwd_id_, grad_output_id_, m_init_id_, grad_logits_id_,
                                              num_iterations_id_, eps_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView grad_output, FTrainStorageView m_init,
                 FTrainStorageView grad_logits, FTrainStorageView num_iterations, FTrainStorageView eps,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus grad_output_status    = ftrainArgsSetTensor(args, grad_output_id_, grad_output);
        const FTrainStatus m_init_status         = ftrainArgsSetTensor(args, m_init_id_, m_init);
        const FTrainStatus grad_logits_status    = ftrainArgsSetTensor(args, grad_logits_id_, grad_logits);
        const FTrainStatus num_iterations_status = ftrainArgsSetTensor(args, num_iterations_id_, num_iterations);
        const FTrainStatus eps_status            = ftrainArgsSetTensor(args, eps_id_, eps);
        const FTrainStatus op_status             = ftrainArgsSetSinkhornBwd(args, sinkhorn_bwd_id_, compute_type);
        EXPECT_EQ(grad_output_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(m_init_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_logits_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(num_iterations_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return grad_output_status == FTRAIN_STATUS_SUCCESS && m_init_status == FTRAIN_STATUS_SUCCESS &&
               grad_logits_status == FTRAIN_STATUS_SUCCESS && num_iterations_status == FTRAIN_STATUS_SUCCESS &&
               eps_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId grad_output_id_{};
    FTrainTensorId m_init_id_{};
    FTrainTensorId grad_logits_id_{};
    FTrainTensorId num_iterations_id_{};
    FTrainTensorId eps_id_{};
    FTrainSinkhornBwdOpId sinkhorn_bwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SINKHORN_BWD_STAGED_FIXTURE_HPP_
