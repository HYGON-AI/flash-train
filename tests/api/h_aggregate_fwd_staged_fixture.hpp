// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_H_AGGREGATE_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_H_AGGREGATE_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class HAggregateFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &aggregated_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &h_pre_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &x_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddHAggregateFwd(pattern_, &h_aggregate_fwd_id_, x_id_, h_pre_id_, aggregated_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView x, FTrainStorageView h_pre, FTrainStorageView aggregated,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus x_status          = ftrainArgsSetTensor(args, x_id_, x);
        const FTrainStatus h_pre_status      = ftrainArgsSetTensor(args, h_pre_id_, h_pre);
        const FTrainStatus aggregated_status = ftrainArgsSetTensor(args, aggregated_id_, aggregated);
        const FTrainStatus op_status         = ftrainArgsSetHAggregateFwd(args, h_aggregate_fwd_id_, compute_type);
        EXPECT_EQ(x_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(h_pre_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(aggregated_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return x_status == FTRAIN_STATUS_SUCCESS && h_pre_status == FTRAIN_STATUS_SUCCESS &&
               aggregated_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId x_id_{};
    FTrainTensorId h_pre_id_{};
    FTrainTensorId aggregated_id_{};
    FTrainHAggregateFwdOpId h_aggregate_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_H_AGGREGATE_FWD_STAGED_FIXTURE_HPP_
