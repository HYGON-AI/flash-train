// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_DENSE_INDEXER_SCORE_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_DENSE_INDEXER_SCORE_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class DenseIndexerScoreStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &sm_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ratio_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &denom_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &w_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddDenseIndexerScore(pattern_, &dense_indexer_score_id_, q_id_, k_id_, w_id_, out_id_,
                                                    denom_id_, ratio_id_, sm_scale_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView k, FTrainStorageView w, FTrainStorageView out,
                 FTrainStorageView denom, FTrainStorageView ratio, FTrainStorageView sm_scale,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status        = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus k_status        = ftrainArgsSetTensor(args, k_id_, k);
        const FTrainStatus w_status        = ftrainArgsSetTensor(args, w_id_, w);
        const FTrainStatus out_status      = ftrainArgsSetTensor(args, out_id_, out);
        const FTrainStatus denom_status    = ftrainArgsSetTensor(args, denom_id_, denom);
        const FTrainStatus ratio_status    = ftrainArgsSetTensor(args, ratio_id_, ratio);
        const FTrainStatus sm_scale_status = ftrainArgsSetTensor(args, sm_scale_id_, sm_scale);
        const FTrainStatus op_status = ftrainArgsSetDenseIndexerScore(args, dense_indexer_score_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(w_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(denom_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ratio_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(sm_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && k_status == FTRAIN_STATUS_SUCCESS &&
               w_status == FTRAIN_STATUS_SUCCESS && out_status == FTRAIN_STATUS_SUCCESS &&
               denom_status == FTRAIN_STATUS_SUCCESS && ratio_status == FTRAIN_STATUS_SUCCESS &&
               sm_scale_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId k_id_{};
    FTrainTensorId w_id_{};
    FTrainTensorId out_id_{};
    FTrainTensorId denom_id_{};
    FTrainTensorId ratio_id_{};
    FTrainTensorId sm_scale_id_{};
    FTrainDenseIndexerScoreOpId dense_indexer_score_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_DENSE_INDEXER_SCORE_STAGED_FIXTURE_HPP_
