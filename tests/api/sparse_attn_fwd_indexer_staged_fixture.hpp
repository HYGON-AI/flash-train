// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SPARSE_ATTN_FWD_INDEXER_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SPARSE_ATTN_FWD_INDEXER_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SparseAttnFwdIndexerStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &indexer_topk_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &softmax_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &lse_indexer_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &lse_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &max_logits_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_length_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &attn_sink_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(
            ftrainPatternAddSparseAttnFwdIndexer(pattern_, &sparse_attn_fwd_indexer_id_, q_id_, kv_id_, attn_sink_id_,
                                                 topk_indices_id_, topk_length_id_, out_id_, max_logits_id_, lse_id_,
                                                 lse_indexer_id_, softmax_scale_id_, indexer_topk_id_),
            FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView kv, FTrainStorageView attn_sink,
                 FTrainStorageView topk_indices, FTrainStorageView topk_length, FTrainStorageView out,
                 FTrainStorageView max_logits, FTrainStorageView lse, FTrainStorageView lse_indexer,
                 FTrainStorageView softmax_scale, FTrainStorageView indexer_topk,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status             = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus kv_status            = ftrainArgsSetTensor(args, kv_id_, kv);
        const FTrainStatus attn_sink_status     = ftrainArgsSetTensor(args, attn_sink_id_, attn_sink);
        const FTrainStatus topk_indices_status  = ftrainArgsSetTensor(args, topk_indices_id_, topk_indices);
        const FTrainStatus topk_length_status   = ftrainArgsSetTensor(args, topk_length_id_, topk_length);
        const FTrainStatus out_status           = ftrainArgsSetTensor(args, out_id_, out);
        const FTrainStatus max_logits_status    = ftrainArgsSetTensor(args, max_logits_id_, max_logits);
        const FTrainStatus lse_status           = ftrainArgsSetTensor(args, lse_id_, lse);
        const FTrainStatus lse_indexer_status   = ftrainArgsSetTensor(args, lse_indexer_id_, lse_indexer);
        const FTrainStatus softmax_scale_status = ftrainArgsSetTensor(args, softmax_scale_id_, softmax_scale);
        const FTrainStatus indexer_topk_status  = ftrainArgsSetTensor(args, indexer_topk_id_, indexer_topk);
        const FTrainStatus op_status =
            ftrainArgsSetSparseAttnFwdIndexer(args, sparse_attn_fwd_indexer_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(attn_sink_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_length_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(max_logits_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(lse_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(lse_indexer_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(softmax_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(indexer_topk_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && kv_status == FTRAIN_STATUS_SUCCESS &&
               attn_sink_status == FTRAIN_STATUS_SUCCESS && topk_indices_status == FTRAIN_STATUS_SUCCESS &&
               topk_length_status == FTRAIN_STATUS_SUCCESS && out_status == FTRAIN_STATUS_SUCCESS &&
               max_logits_status == FTRAIN_STATUS_SUCCESS && lse_status == FTRAIN_STATUS_SUCCESS &&
               lse_indexer_status == FTRAIN_STATUS_SUCCESS && softmax_scale_status == FTRAIN_STATUS_SUCCESS &&
               indexer_topk_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId kv_id_{};
    FTrainTensorId attn_sink_id_{};
    FTrainTensorId topk_indices_id_{};
    FTrainTensorId topk_length_id_{};
    FTrainTensorId out_id_{};
    FTrainTensorId max_logits_id_{};
    FTrainTensorId lse_id_{};
    FTrainTensorId lse_indexer_id_{};
    FTrainTensorId softmax_scale_id_{};
    FTrainTensorId indexer_topk_id_{};
    FTrainSparseAttnFwdIndexerOpId sparse_attn_fwd_indexer_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SPARSE_ATTN_FWD_INDEXER_STAGED_FIXTURE_HPP_
