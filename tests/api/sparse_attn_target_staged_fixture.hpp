// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SPARSE_ATTN_TARGET_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SPARSE_ATTN_TARGET_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SparseAttnTargetStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_global_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &softmax_scale_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &target_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &lse_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &k_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &q_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(
            ftrainPatternAddSparseAttnTarget(pattern_, &sparse_attn_target_id_, q_id_, k_id_, lse_id_, topk_indices_id_,
                                             target_id_, softmax_scale_id_, topk_indices_global_id_),
            FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                 FTrainStorageView topk_indices, FTrainStorageView target, FTrainStorageView softmax_scale,
                 FTrainStorageView topk_indices_global, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus q_status             = ftrainArgsSetTensor(args, q_id_, q);
        const FTrainStatus k_status             = ftrainArgsSetTensor(args, k_id_, k);
        const FTrainStatus lse_status           = ftrainArgsSetTensor(args, lse_id_, lse);
        const FTrainStatus topk_indices_status  = ftrainArgsSetTensor(args, topk_indices_id_, topk_indices);
        const FTrainStatus target_status        = ftrainArgsSetTensor(args, target_id_, target);
        const FTrainStatus softmax_scale_status = ftrainArgsSetTensor(args, softmax_scale_id_, softmax_scale);
        const FTrainStatus topk_indices_global_status =
            ftrainArgsSetTensor(args, topk_indices_global_id_, topk_indices_global);
        const FTrainStatus op_status = ftrainArgsSetSparseAttnTarget(args, sparse_attn_target_id_, compute_type);
        EXPECT_EQ(q_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(k_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(lse_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(target_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(softmax_scale_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_indices_global_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return q_status == FTRAIN_STATUS_SUCCESS && k_status == FTRAIN_STATUS_SUCCESS &&
               lse_status == FTRAIN_STATUS_SUCCESS && topk_indices_status == FTRAIN_STATUS_SUCCESS &&
               target_status == FTRAIN_STATUS_SUCCESS && softmax_scale_status == FTRAIN_STATUS_SUCCESS &&
               topk_indices_global_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId q_id_{};
    FTrainTensorId k_id_{};
    FTrainTensorId lse_id_{};
    FTrainTensorId topk_indices_id_{};
    FTrainTensorId target_id_{};
    FTrainTensorId softmax_scale_id_{};
    FTrainTensorId topk_indices_global_id_{};
    FTrainSparseAttnTargetOpId sparse_attn_target_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SPARSE_ATTN_TARGET_STAGED_FIXTURE_HPP_
