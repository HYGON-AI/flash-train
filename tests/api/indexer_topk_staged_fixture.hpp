// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_INDEXER_TOPK_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_INDEXER_TOPK_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class IndexerTopKStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &order_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &next_n_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &top_k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &seq_lens_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &input_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddIndexerTopK(pattern_, &indexer_topk_id_, input_values_id_, seq_lens_id_,
                                              out_indices_id_, out_values_id_, top_k_id_, next_n_id_, order_values_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView input_values, FTrainStorageView seq_lens,
                 FTrainStorageView out_indices, FTrainStorageView out_values, FTrainStorageView top_k,
                 FTrainStorageView next_n, FTrainStorageView order_values) {
        const FTrainStatus input_values_status = ftrainArgsSetTensor(args, input_values_id_, input_values);
        const FTrainStatus seq_lens_status     = ftrainArgsSetTensor(args, seq_lens_id_, seq_lens);
        const FTrainStatus out_indices_status  = ftrainArgsSetTensor(args, out_indices_id_, out_indices);
        const FTrainStatus out_values_status   = ftrainArgsSetTensor(args, out_values_id_, out_values);
        const FTrainStatus top_k_status        = ftrainArgsSetTensor(args, top_k_id_, top_k);
        const FTrainStatus next_n_status       = ftrainArgsSetTensor(args, next_n_id_, next_n);
        const FTrainStatus order_values_status = ftrainArgsSetTensor(args, order_values_id_, order_values);
        const FTrainStatus op_status           = ftrainArgsSetIndexerTopK(args, indexer_topk_id_);
        EXPECT_EQ(input_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(seq_lens_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(top_k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(next_n_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(order_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return input_values_status == FTRAIN_STATUS_SUCCESS && seq_lens_status == FTRAIN_STATUS_SUCCESS &&
               out_indices_status == FTRAIN_STATUS_SUCCESS && out_values_status == FTRAIN_STATUS_SUCCESS &&
               top_k_status == FTRAIN_STATUS_SUCCESS && next_n_status == FTRAIN_STATUS_SUCCESS &&
               order_values_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId input_values_id_{};
    FTrainTensorId seq_lens_id_{};
    FTrainTensorId out_indices_id_{};
    FTrainTensorId out_values_id_{};
    FTrainTensorId top_k_id_{};
    FTrainTensorId next_n_id_{};
    FTrainTensorId order_values_id_{};
    FTrainIndexerTopKOpId indexer_topk_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_INDEXER_TOPK_STAGED_FIXTURE_HPP_
