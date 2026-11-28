// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_H_AGGREGATE_BWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_H_AGGREGATE_BWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class HAggregateBwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_h_pre_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_pre_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_aggregated_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddHAggregateBwd(pattern_, &h_aggregate_bwd_id_, grad_aggregated_id_, x_id_, h_pre_id_,
                                                grad_x_id_, grad_h_pre_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView grad_aggregated, FTrainStorageView x, FTrainStorageView h_pre,
                 FTrainStorageView grad_x, FTrainStorageView grad_h_pre,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus grad_aggregated_status = ftrainArgsSetTensor(args, grad_aggregated_id_, grad_aggregated);
        const FTrainStatus x_status               = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus h_pre_status           = ftrainArgsSetTensor(args, h_pre_id_, h_pre);
        const FTrainStatus grad_x_status          = ftrainArgsSetTensor(args, grad_x_id_, grad_x);
        const FTrainStatus grad_h_pre_status      = ftrainArgsSetTensor(args, grad_h_pre_id_, grad_h_pre);
        const FTrainStatus op_status              = ftrainArgsSetHAggregateBwd(args, h_aggregate_bwd_id_, compute_type);
        EXPECT_EQ(grad_aggregated_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_pre_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_h_pre_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return grad_aggregated_status == FTRAIN_STATUS_SUCCESS && x_status == FTRAIN_STATUS_SUCCESS &&
               h_pre_status == FTRAIN_STATUS_SUCCESS && grad_x_status == FTRAIN_STATUS_SUCCESS &&
               grad_h_pre_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId grad_aggregated_id_{};
    FTrainTensorId x_id_{};
    FTrainTensorId h_pre_id_{};
    FTrainTensorId grad_x_id_{};
    FTrainTensorId grad_h_pre_id_{};
    FTrainHAggregateBwdOpId h_aggregate_bwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_H_AGGREGATE_BWD_STAGED_FIXTURE_HPP_
