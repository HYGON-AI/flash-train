// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_INDEXER_FWD_THD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_INDEXER_FWD_THD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class IndexerFwdThdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &max_seqlen_kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &sm_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ratio_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &scores_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_causal_offsets_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &w_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddIndexerFwdThd(pattern_, &indexer_fwd_thd_id_, q_id_, k_id_, w_id_, cu_seqlens_q_id_,
                                                cu_seqlens_kv_id_, q_causal_offsets_id_, scores_id_, ratio_id_,
                                                sm_scale_id_, max_seqlen_kv_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                 FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv, FTrainStorageView q_causal_offsets,
                 FTrainStorageView scores, FTrainStorageView ratio, FTrainStorageView sm_scale,
                 FTrainStorageView max_seqlen_kv, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status                = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus k_status                = ftrainArgsSetTensor(args, k_id_, k);
        const FTrainStatus w_status                = ftrainArgsSetTensor(args, w_id_, w);
        const FTrainStatus cu_seqlens_q_status     = ftrainArgsSetTensor(args, cu_seqlens_q_id_, cu_seqlens_q);
        const FTrainStatus cu_seqlens_kv_status    = ftrainArgsSetTensor(args, cu_seqlens_kv_id_, cu_seqlens_kv);
        const FTrainStatus q_causal_offsets_status = ftrainArgsSetTensor(args, q_causal_offsets_id_, q_causal_offsets);
        const FTrainStatus scores_status           = ftrainArgsSetTensor(args, scores_id_, scores);
        const FTrainStatus ratio_status            = ftrainArgsSetTensor(args, ratio_id_, ratio);
        const FTrainStatus sm_scale_status         = ftrainArgsSetTensor(args, sm_scale_id_, sm_scale);
        const FTrainStatus max_seqlen_kv_status    = ftrainArgsSetTensor(args, max_seqlen_kv_id_, max_seqlen_kv);
        const FTrainStatus op_status = ftrainArgsSetIndexerFwdThd(args, indexer_fwd_thd_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(w_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(q_causal_offsets_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(scores_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ratio_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(sm_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(max_seqlen_kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && k_status == FTRAIN_STATUS_SUCCESS &&
               w_status == FTRAIN_STATUS_SUCCESS && cu_seqlens_q_status == FTRAIN_STATUS_SUCCESS &&
               cu_seqlens_kv_status == FTRAIN_STATUS_SUCCESS && q_causal_offsets_status == FTRAIN_STATUS_SUCCESS &&
               scores_status == FTRAIN_STATUS_SUCCESS && ratio_status == FTRAIN_STATUS_SUCCESS &&
               sm_scale_status == FTRAIN_STATUS_SUCCESS && max_seqlen_kv_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId k_id_{};
    FTrainTensorId w_id_{};
    FTrainTensorId cu_seqlens_q_id_{};
    FTrainTensorId cu_seqlens_kv_id_{};
    FTrainTensorId q_causal_offsets_id_{};
    FTrainTensorId scores_id_{};
    FTrainTensorId ratio_id_{};
    FTrainTensorId sm_scale_id_{};
    FTrainTensorId max_seqlen_kv_id_{};
    FTrainIndexerFwdThdOpId indexer_fwd_thd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_INDEXER_FWD_THD_STAGED_FIXTURE_HPP_
