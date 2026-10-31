// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/family/gemm/finder.hpp"
#include "flash_train/family/gemm/family.hpp"
#include "flash_train/family/compactify/family.hpp"
#include "flash_train/family/compactify/finder.hpp"
#include "flash_train/family/csa_compressor_bwd/family.hpp"
#include "flash_train/family/csa_compressor_bwd/finder.hpp"
#include "flash_train/family/csa_compressor_fwd/family.hpp"
#include "flash_train/family/csa_compressor_fwd/finder.hpp"
#include "flash_train/family/dense_attn_score/family.hpp"
#include "flash_train/family/dense_attn_score/finder.hpp"
#include "flash_train/family/dense_attn_score_thd/family.hpp"
#include "flash_train/family/dense_attn_score_thd/finder.hpp"
#include "flash_train/family/dense_indexer_score/family.hpp"
#include "flash_train/family/dense_indexer_score/finder.hpp"
#include "flash_train/family/dense_indexer_score_thd/family.hpp"
#include "flash_train/family/dense_indexer_score_thd/finder.hpp"
#include "flash_train/family/indexer_bwd_dense/family.hpp"
#include "flash_train/family/indexer_bwd_dense/finder.hpp"
#include "flash_train/family/indexer_bwd_dense_thd/family.hpp"
#include "flash_train/family/indexer_bwd_dense_thd/finder.hpp"
#include "flash_train/family/indexer_bwd_sparse/family.hpp"
#include "flash_train/family/indexer_bwd_sparse/finder.hpp"
#include "flash_train/family/indexer_fwd/family.hpp"
#include "flash_train/family/indexer_fwd/finder.hpp"
#include "flash_train/family/indexer_fwd_thd/family.hpp"
#include "flash_train/family/indexer_fwd_thd/finder.hpp"
#include "flash_train/family/indexer_topk/family.hpp"
#include "flash_train/family/indexer_topk/finder.hpp"
#include "flash_train/family/indexer_topk_thd/family.hpp"
#include "flash_train/family/indexer_topk_thd/finder.hpp"
#include "flash_train/family/sparse_attn_bwd/family.hpp"
#include "flash_train/family/sparse_attn_bwd/finder.hpp"
#include "flash_train/family/sparse_attn_fwd/family.hpp"
#include "flash_train/family/sparse_attn_fwd/finder.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/family.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/finder.hpp"
#include "flash_train/family/sparse_attn_target/family.hpp"
#include "flash_train/family/sparse_attn_target/finder.hpp"
#include "flash_train/family/sparse_indexer_predict/family.hpp"
#include "flash_train/family/sparse_indexer_predict/finder.hpp"
#include "flash_train/family/h_aggregate_bwd/family.hpp"
#include "flash_train/family/h_aggregate_bwd/finder.hpp"
#include "flash_train/family/h_aggregate_fwd/family.hpp"
#include "flash_train/family/h_aggregate_fwd/finder.hpp"
#include "flash_train/family/h_post_bda_bwd/family.hpp"
#include "flash_train/family/h_post_bda_bwd/finder.hpp"
#include "flash_train/family/h_post_bda_fwd/family.hpp"
#include "flash_train/family/h_post_bda_fwd/finder.hpp"
#include "flash_train/family/proj_rms_bwd/family.hpp"
#include "flash_train/family/proj_rms_bwd/finder.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/family.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/finder.hpp"
#include "flash_train/family/proj_rms_compute_h_fwd/family.hpp"
#include "flash_train/family/proj_rms_compute_h_fwd/finder.hpp"
#include "flash_train/family/proj_rms_fwd/family.hpp"
#include "flash_train/family/proj_rms_fwd/finder.hpp"
#include "flash_train/family/sinkhorn_bwd/family.hpp"
#include "flash_train/family/sinkhorn_bwd/finder.hpp"
#include "flash_train/family/sinkhorn_fwd/family.hpp"
#include "flash_train/family/sinkhorn_fwd/finder.hpp"
#include "flash_train/engine.hpp"

namespace ftrain {

OpsEngineBase::OpsEngineBase(const Pattern& supported_pattern) : supported_pattern_(supported_pattern) {}

OpsEngineBase::~OpsEngineBase() = default;

Plan OpsEngineBase::createPlan(const Args& args) const {
    const FTrainDeviceId device_id = getCurrentDeviceId();
    return Plan(createPrimitives(args, Constraints{device_id}), device_id);
}

// The built-in engine catalog and the single place that enumerates the
// concrete operator families. Families add themselves under family/<op>/
// and register by constructing their engine here.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines() {
    std::vector<std::shared_ptr<OpsEngineBase>> engines;
    engines.push_back(std::make_shared<OpsEngine<GemmFamily, RegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerFwdFamily, IndexerFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerFwdThdFamily, IndexerFwdThdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<DenseIndexerScoreFamily, DenseIndexerScoreRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<DenseIndexerScoreThdFamily, DenseIndexerScoreThdRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<SparseIndexerPredictFamily, SparseIndexerPredictRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<SparseAttnFwdFamily, SparseAttnFwdRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<SparseAttnFwdIndexerFamily, SparseAttnFwdIndexerRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<SparseAttnBwdFamily, SparseAttnBwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerTopKFamily, IndexerTopKRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<CompactifyFamily, CompactifyRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<DenseAttnScoreFamily, DenseAttnScoreRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<DenseAttnScoreThdFamily, DenseAttnScoreThdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<SparseAttnTargetFamily, SparseAttnTargetRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerTopKThdFamily, IndexerTopKThdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerBwdSparseFamily, IndexerBwdSparseRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<IndexerBwdDenseFamily, IndexerBwdDenseRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<IndexerBwdDenseThdFamily, IndexerBwdDenseThdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<CSACompressorFwdFamily, CSACompressorFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<CSACompressorBwdFamily, CSACompressorBwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<SinkhornFwdFamily, SinkhornFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<SinkhornBwdFamily, SinkhornBwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<HAggregateFwdFamily, HAggregateFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<HAggregateBwdFamily, HAggregateBwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<HPostBdaFwdFamily, HPostBdaFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<HPostBdaBwdFamily, HPostBdaBwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<ProjRmsFwdFamily, ProjRmsFwdRegistrationOrderFinder>>());
    engines.push_back(std::make_shared<OpsEngine<ProjRmsBwdFamily, ProjRmsBwdRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<ProjRmsComputeHFwdFamily, ProjRmsComputeHFwdRegistrationOrderFinder>>());
    engines.push_back(
        std::make_shared<OpsEngine<ProjRmsComputeHBwdFamily, ProjRmsComputeHBwdRegistrationOrderFinder>>());
    return engines;
}

}  // namespace ftrain
