// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_DENSE_ATTN_SCORE_THD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_DENSE_ATTN_SCORE_THD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class DenseAttnScoreThdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &max_seqlen_kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &softmax_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ratio_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &denom_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &out_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_causal_offsets_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &lse_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddDenseAttnScoreThd(pattern_, &dense_attn_score_thd_id_, q_id_, k_id_, lse_id_,
                                                    cu_seqlens_q_id_, cu_seqlens_kv_id_, q_causal_offsets_id_, out_id_,
                                                    denom_id_, ratio_id_, softmax_scale_id_, max_seqlen_kv_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                 FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv, FTrainStorageView q_causal_offsets,
                 FTrainStorageView out, FTrainStorageView denom, FTrainStorageView ratio,
                 FTrainStorageView softmax_scale, FTrainStorageView max_seqlen_kv,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status                = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus k_status                = ftrainArgsSetTensor(args, k_id_, k);
        const FTrainStatus lse_status              = ftrainArgsSetTensor(args, lse_id_, lse);
        const FTrainStatus cu_seqlens_q_status     = ftrainArgsSetTensor(args, cu_seqlens_q_id_, cu_seqlens_q);
        const FTrainStatus cu_seqlens_kv_status    = ftrainArgsSetTensor(args, cu_seqlens_kv_id_, cu_seqlens_kv);
        const FTrainStatus q_causal_offsets_status = ftrainArgsSetTensor(args, q_causal_offsets_id_, q_causal_offsets);
        const FTrainStatus out_status              = ftrainArgsSetTensor(args, out_id_, out);
        const FTrainStatus denom_status            = ftrainArgsSetTensor(args, denom_id_, denom);
        const FTrainStatus ratio_status            = ftrainArgsSetTensor(args, ratio_id_, ratio);
        const FTrainStatus softmax_scale_status    = ftrainArgsSetTensor(args, softmax_scale_id_, softmax_scale);
        const FTrainStatus max_seqlen_kv_status    = ftrainArgsSetTensor(args, max_seqlen_kv_id_, max_seqlen_kv);
        const FTrainStatus op_status = ftrainArgsSetDenseAttnScoreThd(args, dense_attn_score_thd_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(lse_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(q_causal_offsets_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(out_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(denom_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ratio_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(softmax_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(max_seqlen_kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && k_status == FTRAIN_STATUS_SUCCESS &&
               lse_status == FTRAIN_STATUS_SUCCESS && cu_seqlens_q_status == FTRAIN_STATUS_SUCCESS &&
               cu_seqlens_kv_status == FTRAIN_STATUS_SUCCESS && q_causal_offsets_status == FTRAIN_STATUS_SUCCESS &&
               out_status == FTRAIN_STATUS_SUCCESS && denom_status == FTRAIN_STATUS_SUCCESS &&
               ratio_status == FTRAIN_STATUS_SUCCESS && softmax_scale_status == FTRAIN_STATUS_SUCCESS &&
               max_seqlen_kv_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId k_id_{};
    FTrainTensorId lse_id_{};
    FTrainTensorId cu_seqlens_q_id_{};
    FTrainTensorId cu_seqlens_kv_id_{};
    FTrainTensorId q_causal_offsets_id_{};
    FTrainTensorId out_id_{};
    FTrainTensorId denom_id_{};
    FTrainTensorId ratio_id_{};
    FTrainTensorId softmax_scale_id_{};
    FTrainTensorId max_seqlen_kv_id_{};
    FTrainDenseAttnScoreThdOpId dense_attn_score_thd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_DENSE_ATTN_SCORE_THD_STAGED_FIXTURE_HPP_
