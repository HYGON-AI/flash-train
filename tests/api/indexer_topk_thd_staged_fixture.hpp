// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_INDEXER_TOPK_THD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_INDEXER_TOPK_THD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class IndexerTopKThdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &order_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &top_k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ratio_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_causal_offsets_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &input_values_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddIndexerTopKThd(pattern_, &indexer_topk_thd_id_, input_values_id_, cu_seqlens_q_id_,
                                                 cu_seqlens_kv_id_, q_causal_offsets_id_, out_indices_id_,
                                                 out_values_id_, ratio_id_, top_k_id_, order_values_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView input_values, FTrainStorageView cu_seqlens_q,
                 FTrainStorageView cu_seqlens_kv, FTrainStorageView q_causal_offsets, FTrainStorageView out_indices,
                 FTrainStorageView out_values, FTrainStorageView ratio, FTrainStorageView top_k,
                 FTrainStorageView order_values) {
        const FTrainStatus input_values_status     = ftrainArgsSetTensor(args, input_values_id_, input_values);
        const FTrainStatus cu_seqlens_q_status     = ftrainArgsSetTensor(args, cu_seqlens_q_id_, cu_seqlens_q);
        const FTrainStatus cu_seqlens_kv_status    = ftrainArgsSetTensor(args, cu_seqlens_kv_id_, cu_seqlens_kv);
        const FTrainStatus q_causal_offsets_status = ftrainArgsSetTensor(args, q_causal_offsets_id_, q_causal_offsets);
        const FTrainStatus out_indices_status      = ftrainArgsSetTensor(args, out_indices_id_, out_indices);
        const FTrainStatus out_values_status       = ftrainArgsSetTensor(args, out_values_id_, out_values);
        const FTrainStatus ratio_status            = ftrainArgsSetTensor(args, ratio_id_, ratio);
        const FTrainStatus top_k_status            = ftrainArgsSetTensor(args, top_k_id_, top_k);
        const FTrainStatus order_values_status     = ftrainArgsSetTensor(args, order_values_id_, order_values);
        const FTrainStatus op_status               = ftrainArgsSetIndexerTopKThd(args, indexer_topk_thd_id_);
        EXPECT_EQ(input_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(q_causal_offsets_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ratio_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(top_k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(order_values_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return input_values_status == FTRAIN_STATUS_SUCCESS && cu_seqlens_q_status == FTRAIN_STATUS_SUCCESS &&
               cu_seqlens_kv_status == FTRAIN_STATUS_SUCCESS && q_causal_offsets_status == FTRAIN_STATUS_SUCCESS &&
               out_indices_status == FTRAIN_STATUS_SUCCESS && out_values_status == FTRAIN_STATUS_SUCCESS &&
               ratio_status == FTRAIN_STATUS_SUCCESS && top_k_status == FTRAIN_STATUS_SUCCESS &&
               order_values_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId input_values_id_{};
    FTrainTensorId cu_seqlens_q_id_{};
    FTrainTensorId cu_seqlens_kv_id_{};
    FTrainTensorId q_causal_offsets_id_{};
    FTrainTensorId out_indices_id_{};
    FTrainTensorId out_values_id_{};
    FTrainTensorId ratio_id_{};
    FTrainTensorId top_k_id_{};
    FTrainTensorId order_values_id_{};
    FTrainIndexerTopKThdOpId indexer_topk_thd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_INDEXER_TOPK_THD_STAGED_FIXTURE_HPP_
