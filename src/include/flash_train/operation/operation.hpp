// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_OPERATION_HPP_
#define FTRAIN_OPERATION_OPERATION_HPP_

#include <variant>

#include "flash_train/operation/gemm.hpp"
#include "flash_train/operation/compactify.hpp"
#include "flash_train/operation/csa_compressor_bwd.hpp"
#include "flash_train/operation/csa_compressor_fwd.hpp"
#include "flash_train/operation/dense_attn_score.hpp"
#include "flash_train/operation/dense_attn_score_thd.hpp"
#include "flash_train/operation/dense_indexer_score.hpp"
#include "flash_train/operation/dense_indexer_score_thd.hpp"
#include "flash_train/operation/h_aggregate_bwd.hpp"
#include "flash_train/operation/h_aggregate_fwd.hpp"
#include "flash_train/operation/h_post_bda_bwd.hpp"
#include "flash_train/operation/h_post_bda_fwd.hpp"
#include "flash_train/operation/indexer_bwd_dense.hpp"
#include "flash_train/operation/indexer_bwd_dense_thd.hpp"
#include "flash_train/operation/indexer_bwd_sparse.hpp"
#include "flash_train/operation/indexer_fwd.hpp"
#include "flash_train/operation/indexer_fwd_thd.hpp"
#include "flash_train/operation/indexer_topk.hpp"
#include "flash_train/operation/indexer_topk_thd.hpp"
#include "flash_train/operation/proj_rms_bwd.hpp"
#include "flash_train/operation/proj_rms_compute_h_bwd.hpp"
#include "flash_train/operation/proj_rms_compute_h_fwd.hpp"
#include "flash_train/operation/proj_rms_fwd.hpp"
#include "flash_train/operation/sinkhorn_bwd.hpp"
#include "flash_train/operation/sinkhorn_fwd.hpp"
#include "flash_train/operation/sparse_attn_bwd.hpp"
#include "flash_train/operation/sparse_attn_fwd.hpp"
#include "flash_train/operation/sparse_attn_fwd_indexer.hpp"
#include "flash_train/operation/sparse_attn_target.hpp"
#include "flash_train/operation/sparse_indexer_predict.hpp"

namespace ftrain {

using OperationValue =
    std::variant<GemmAttributes, GroupedABCDGemmAttributes, GroupedBCDGemmAttributes, GroupedABGemmAttributes,

                 IndexerFwdAttributes, IndexerFwdThdAttributes, DenseIndexerScoreAttributes,
                 DenseIndexerScoreThdAttributes, DenseAttnScoreAttributes, DenseAttnScoreThdAttributes,
                 SparseIndexerPredictAttributes, SparseAttnTargetAttributes, SparseAttnFwdAttributes,
                 SparseAttnFwdIndexerAttributes, SparseAttnBwdAttributes, IndexerBwdSparseAttributes,
                 IndexerBwdDenseAttributes, IndexerBwdDenseThdAttributes, IndexerTopKAttributes,
                 IndexerTopKThdAttributes, CompactifyAttributes, CSACompressorFwdAttributes, CSACompressorBwdAttributes,

                 SinkhornFwdAttributes, SinkhornBwdAttributes, HAggregateFwdAttributes, HAggregateBwdAttributes,
                 HPostBdaFwdAttributes, HPostBdaBwdAttributes, ProjRmsFwdAttributes, ProjRmsBwdAttributes,
                 ProjRmsComputeHFwdAttributes, ProjRmsComputeHBwdAttributes>;

}  // namespace ftrain

#endif
