// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_CSA_COMPRESSOR_FWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_CSA_COMPRESSOR_FWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class CSACompressorFwdStagedTest : public StagedApiTest {
  protected:
    void SetUp() override {
        StagedApiTest::SetUp();
        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &total_comp_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &coff_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &head_dim_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ratio_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &compressed_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_comp_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &cu_seqlens_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &ape_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &score_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &kv_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddCSACompressorFwd(pattern_, &csa_compressor_fwd_id_, kv_id_, score_id_, ape_id_,
                                                   cu_seqlens_id_, cu_seqlens_comp_id_, compressed_id_, ratio_id_,
                                                   head_dim_id_, coff_id_, total_comp_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    bool setArgs(FTrainArgs args, FTrainStorageView kv, FTrainStorageView score, FTrainStorageView ape,
                 FTrainStorageView cu_seqlens, FTrainStorageView cu_seqlens_comp, FTrainStorageView compressed,
                 FTrainStorageView ratio, FTrainStorageView head_dim, FTrainStorageView coff,
                 FTrainStorageView total_comp, FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_BF16) {
        const FTrainStatus kv_status              = ftrainArgsSetTensor(args, kv_id_, kv);
        const FTrainStatus score_status           = ftrainArgsSetTensor(args, score_id_, score);
        const FTrainStatus ape_status             = ftrainArgsSetTensor(args, ape_id_, ape);
        const FTrainStatus cu_seqlens_status      = ftrainArgsSetTensor(args, cu_seqlens_id_, cu_seqlens);
        const FTrainStatus cu_seqlens_comp_status = ftrainArgsSetTensor(args, cu_seqlens_comp_id_, cu_seqlens_comp);
        const FTrainStatus compressed_status      = ftrainArgsSetTensor(args, compressed_id_, compressed);
        const FTrainStatus ratio_status           = ftrainArgsSetTensor(args, ratio_id_, ratio);
        const FTrainStatus head_dim_status        = ftrainArgsSetTensor(args, head_dim_id_, head_dim);
        const FTrainStatus coff_status            = ftrainArgsSetTensor(args, coff_id_, coff);
        const FTrainStatus total_comp_status      = ftrainArgsSetTensor(args, total_comp_id_, total_comp);
        const FTrainStatus op_status = ftrainArgsSetCSACompressorFwd(args, csa_compressor_fwd_id_, compute_type);
        EXPECT_EQ(kv_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(score_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ape_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(cu_seqlens_comp_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(compressed_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ratio_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(head_dim_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(coff_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(total_comp_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return kv_status == FTRAIN_STATUS_SUCCESS && score_status == FTRAIN_STATUS_SUCCESS &&
               ape_status == FTRAIN_STATUS_SUCCESS && cu_seqlens_status == FTRAIN_STATUS_SUCCESS &&
               cu_seqlens_comp_status == FTRAIN_STATUS_SUCCESS && compressed_status == FTRAIN_STATUS_SUCCESS &&
               ratio_status == FTRAIN_STATUS_SUCCESS && head_dim_status == FTRAIN_STATUS_SUCCESS &&
               coff_status == FTRAIN_STATUS_SUCCESS && total_comp_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    FTrainTensorId kv_id_{};
    FTrainTensorId score_id_{};
    FTrainTensorId ape_id_{};
    FTrainTensorId cu_seqlens_id_{};
    FTrainTensorId cu_seqlens_comp_id_{};
    FTrainTensorId compressed_id_{};
    FTrainTensorId ratio_id_{};
    FTrainTensorId head_dim_id_{};
    FTrainTensorId coff_id_{};
    FTrainTensorId total_comp_id_{};
    FTrainCSACompressorFwdOpId csa_compressor_fwd_id_{};
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_CSA_COMPRESSOR_FWD_STAGED_FIXTURE_HPP_
