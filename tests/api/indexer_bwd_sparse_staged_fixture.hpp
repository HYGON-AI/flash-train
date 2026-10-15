// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_INDEXER_BWD_SPARSE_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_INDEXER_BWD_SPARSE_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class IndexerBwdSparseStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_global_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_loss_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &loss_coeff_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &sm_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &dk_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &dw_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &dq_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &grad_attn_score_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &index_score_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &attn_score_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &index_k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &weights_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &index_q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddIndexerBwdSparse(pattern_, &indexer_bwd_sparse_id_, index_q_id_, weights_id_,
                                                   index_k_id_, attn_score_id_, index_score_id_, topk_indices_id_,
                                                   grad_attn_score_id_, dq_id_, dw_id_, dk_id_, sm_scale_id_,
                                                   loss_coeff_id_, grad_loss_id_, topk_indices_global_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView index_q, FTrainStorageView weights, FTrainStorageView index_k,
                 FTrainStorageView attn_score, FTrainStorageView index_score, FTrainStorageView topk_indices,
                 FTrainStorageView grad_attn_score, FTrainStorageView dq, FTrainStorageView dw, FTrainStorageView dk,
                 FTrainStorageView sm_scale, FTrainStorageView loss_coeff, FTrainStorageView grad_loss,
                 FTrainStorageView topk_indices_global, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus index_q_status         = ftrainArgsSetTensor(args, index_q_id_, index_q);
        const FTrainStatus weights_status         = ftrainArgsSetTensor(args, weights_id_, weights);
        const FTrainStatus index_k_status         = ftrainArgsSetTensor(args, index_k_id_, index_k);
        const FTrainStatus attn_score_status      = ftrainArgsSetTensor(args, attn_score_id_, attn_score);
        const FTrainStatus index_score_status     = ftrainArgsSetTensor(args, index_score_id_, index_score);
        const FTrainStatus topk_indices_status    = ftrainArgsSetTensor(args, topk_indices_id_, topk_indices);
        const FTrainStatus grad_attn_score_status = ftrainArgsSetTensor(args, grad_attn_score_id_, grad_attn_score);
        const FTrainStatus dq_status              = ftrainArgsSetTensor(args, dq_id_, dq);
        const FTrainStatus dw_status              = ftrainArgsSetTensor(args, dw_id_, dw);
        const FTrainStatus dk_status              = ftrainArgsSetTensor(args, dk_id_, dk);
        const FTrainStatus sm_scale_status        = ftrainArgsSetTensor(args, sm_scale_id_, sm_scale);
        const FTrainStatus loss_coeff_status      = ftrainArgsSetTensor(args, loss_coeff_id_, loss_coeff);
        const FTrainStatus grad_loss_status       = ftrainArgsSetTensor(args, grad_loss_id_, grad_loss);
        const FTrainStatus topk_indices_global_status =
            ftrainArgsSetTensor(args, topk_indices_global_id_, topk_indices_global);
        const FTrainStatus op_status = ftrainArgsSetIndexerBwdSparse(args, indexer_bwd_sparse_id_, compute_type);
        EXPECT_EQ(index_q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(weights_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(index_k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(attn_score_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(index_score_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_attn_score_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(dq_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(dw_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(dk_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(sm_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(loss_coeff_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(grad_loss_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_global_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return index_q_status == FTRAIN_STATUS_SUCCESS && weights_status == FTRAIN_STATUS_SUCCESS &&
               index_k_status == FTRAIN_STATUS_SUCCESS && attn_score_status == FTRAIN_STATUS_SUCCESS &&
               index_score_status == FTRAIN_STATUS_SUCCESS && topk_indices_status == FTRAIN_STATUS_SUCCESS &&
               grad_attn_score_status == FTRAIN_STATUS_SUCCESS && dq_status == FTRAIN_STATUS_SUCCESS &&
               dw_status == FTRAIN_STATUS_SUCCESS && dk_status == FTRAIN_STATUS_SUCCESS &&
               sm_scale_status == FTRAIN_STATUS_SUCCESS && loss_coeff_status == FTRAIN_STATUS_SUCCESS &&
               grad_loss_status == FTRAIN_STATUS_SUCCESS && topk_indices_global_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId index_q_id_{};
    FTrainTensorId weights_id_{};
    FTrainTensorId index_k_id_{};
    FTrainTensorId attn_score_id_{};
    FTrainTensorId index_score_id_{};
    FTrainTensorId topk_indices_id_{};
    FTrainTensorId grad_attn_score_id_{};
    FTrainTensorId dq_id_{};
    FTrainTensorId dw_id_{};
    FTrainTensorId dk_id_{};
    FTrainTensorId sm_scale_id_{};
    FTrainTensorId loss_coeff_id_{};
    FTrainTensorId grad_loss_id_{};
    FTrainTensorId topk_indices_global_id_{};
    FTrainIndexerBwdSparseOpId indexer_bwd_sparse_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_INDEXER_BWD_SPARSE_STAGED_FIXTURE_HPP_
