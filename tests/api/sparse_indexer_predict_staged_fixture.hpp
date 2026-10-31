// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SPARSE_INDEXER_PREDICT_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SPARSE_INDEXER_PREDICT_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SparseIndexerPredictStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_global_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &sm_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &predict_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &w_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(
            ftrainPatternAddSparseIndexerPredict(pattern_, &sparse_indexer_predict_id_, q_id_, k_id_, w_id_,
                                                 topk_indices_id_, predict_id_, sm_scale_id_, topk_indices_global_id_),
            FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                 FTrainStorageView topk_indices, FTrainStorageView predict, FTrainStorageView sm_scale,
                 FTrainStorageView topk_indices_global, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status            = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus k_status            = ftrainArgsSetTensor(args, k_id_, k);
        const FTrainStatus w_status            = ftrainArgsSetTensor(args, w_id_, w);
        const FTrainStatus topk_indices_status = ftrainArgsSetTensor(args, topk_indices_id_, topk_indices);
        const FTrainStatus predict_status      = ftrainArgsSetTensor(args, predict_id_, predict);
        const FTrainStatus sm_scale_status     = ftrainArgsSetTensor(args, sm_scale_id_, sm_scale);
        const FTrainStatus topk_indices_global_status =
            ftrainArgsSetTensor(args, topk_indices_global_id_, topk_indices_global);
        const FTrainStatus op_status =
            ftrainArgsSetSparseIndexerPredict(args, sparse_indexer_predict_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(w_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(predict_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(sm_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_global_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && k_status == FTRAIN_STATUS_SUCCESS &&
               w_status == FTRAIN_STATUS_SUCCESS && topk_indices_status == FTRAIN_STATUS_SUCCESS &&
               predict_status == FTRAIN_STATUS_SUCCESS && sm_scale_status == FTRAIN_STATUS_SUCCESS &&
               topk_indices_global_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId k_id_{};
    FTrainTensorId w_id_{};
    FTrainTensorId topk_indices_id_{};
    FTrainTensorId predict_id_{};
    FTrainTensorId sm_scale_id_{};
    FTrainTensorId topk_indices_global_id_{};
    FTrainSparseIndexerPredictOpId sparse_indexer_predict_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SPARSE_INDEXER_PREDICT_STAGED_FIXTURE_HPP_
