// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SINKHORN_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SINKHORN_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SinkhornFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &eps_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &num_iterations_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &m_init_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &output_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &logits_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddSinkhornFwd(pattern_, &sinkhorn_fwd_id_, logits_id_, output_id_, m_init_id_,
                                              num_iterations_id_, eps_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView logits, FTrainStorageView output, FTrainStorageView m_init,
                 FTrainStorageView num_iterations, FTrainStorageView eps,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus logits_status         = ftrainArgsSetTensor(args, logits_id_, logits);
        const FTrainStatus output_status         = ftrainArgsSetTensor(args, output_id_, output);
        const FTrainStatus m_init_status         = ftrainArgsSetTensor(args, m_init_id_, m_init);
        const FTrainStatus num_iterations_status = ftrainArgsSetTensor(args, num_iterations_id_, num_iterations);
        const FTrainStatus eps_status            = ftrainArgsSetTensor(args, eps_id_, eps);
        const FTrainStatus op_status             = ftrainArgsSetSinkhornFwd(args, sinkhorn_fwd_id_, compute_type);
        EXPECT_EQ(logits_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(output_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(m_init_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(num_iterations_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(eps_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return logits_status == FTRAIN_STATUS_SUCCESS && output_status == FTRAIN_STATUS_SUCCESS &&
               m_init_status == FTRAIN_STATUS_SUCCESS && num_iterations_status == FTRAIN_STATUS_SUCCESS &&
               eps_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId logits_id_{};
    FTrainTensorId output_id_{};
    FTrainTensorId m_init_id_{};
    FTrainTensorId num_iterations_id_{};
    FTrainTensorId eps_id_{};
    FTrainSinkhornFwdOpId sinkhorn_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SINKHORN_FWD_STAGED_FIXTURE_HPP_
