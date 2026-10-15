// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_COMPACTIFY_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_COMPACTIFY_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class CompactifyStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &topk_length_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &output_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &indices_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddCompactify(pattern_, &compactify_id_, indices_id_, output_id_, topk_length_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView indices, FTrainStorageView output, FTrainStorageView topk_length) {
        const FTrainStatus indices_status     = ftrainArgsSetTensor(args, indices_id_, indices);
        const FTrainStatus output_status      = ftrainArgsSetTensor(args, output_id_, output);
        const FTrainStatus topk_length_status = ftrainArgsSetTensor(args, topk_length_id_, topk_length);
        const FTrainStatus op_status          = ftrainArgsSetCompactify(args, compactify_id_);
        EXPECT_EQ(indices_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(output_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(topk_length_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return indices_status == FTRAIN_STATUS_SUCCESS && output_status == FTRAIN_STATUS_SUCCESS &&
               topk_length_status == FTRAIN_STATUS_SUCCESS && op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId indices_id_{};
    FTrainTensorId output_id_{};
    FTrainTensorId topk_length_id_{};
    FTrainCompactifyOpId compactify_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_COMPACTIFY_STAGED_FIXTURE_HPP_
