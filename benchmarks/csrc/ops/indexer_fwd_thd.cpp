// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <hip/hip_fp16.h>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct IndexerFwdThdSession final : OpSession {
    IndexerFwdThdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t batch = 1, kv_tokens = a, top_k = std::min<std::int64_t>(a, 32), ratio = 1;
        std::vector<std::int64_t> segments{a};
        if (shape[0] == 8192 && shape[1] == 64 && shape[2] == 128) {
            segments = {4096, 4096};
            ratio    = 4;
        }
        if (shape[0] == 12288 && shape[1] == 8 && shape[2] == 128) {
            segments = {4096, 8192};
            ratio    = 4;
        }
        if (shape[0] == 32768 && shape[1] == 64 && shape[2] == 128) {
            segments = {16384, 16384};
            ratio    = 128;
        }
        if (shape[0] == 8192 && shape[1] == 32 && shape[2] == 128) {
            segments = {4096, 4096};
            ratio    = 1;
        }
        if (shape[0] == 12288 && shape[1] == 4 && shape[2] == 128) {
            segments = {4096, 8192};
            ratio    = 1;
        }
        std::vector<std::int64_t> cu_q{0}, cu_k{0};
        std::int64_t max_k = 0;
        for (auto length : segments) {
            cu_q.push_back(cu_q.back() + length);
            cu_k.push_back(cu_k.back() + length / ratio);
            max_k = std::max(max_k, length / ratio);
        }
        kv_tokens = cu_k.back();
        std::array<FTrainNumericType, 10> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64};
                break;
            default: require(false, "unsupported precision");
        }
        q_dims_ = {a, b, c};
        const std::size_t q_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t q_bytes = q_count * elementBytes(types[0]);
        host_.resize((q_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_count; ++j) { setHostValue(j, 0.25, types[0]); }
        checkHip(hipMalloc(&dq_, q_bytes), "malloc q");
        checkHip(hipMemcpy(dq_, host_.data(), q_bytes, hipMemcpyHostToDevice), "fill q");
        view_q_ = dense(dq_, q_dims_, types[0]);
        k_dims_ = {kv_tokens, 1, c};
        const std::size_t k_count =
            static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(1) * static_cast<std::size_t>(c);
        const std::size_t k_bytes = k_count * elementBytes(types[1]);
        host_.resize((k_bytes + 7) / 8);
        for (std::size_t j = 0; j < k_count; ++j) { setHostValue(j, 0.5, types[1]); }
        checkHip(hipMalloc(&dk_, k_bytes), "malloc k");
        checkHip(hipMemcpy(dk_, host_.data(), k_bytes, hipMemcpyHostToDevice), "fill k");
        view_k_                   = dense(dk_, k_dims_, types[1]);
        w_dims_                   = {a, b};
        const std::size_t w_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t w_bytes = w_count * elementBytes(types[2]);
        host_.resize((w_bytes + 7) / 8);
        for (std::size_t j = 0; j < w_count; ++j) { setHostValue(j, 1, types[2]); }
        checkHip(hipMalloc(&dw_, w_bytes), "malloc w");
        checkHip(hipMemcpy(dw_, host_.data(), w_bytes, hipMemcpyHostToDevice), "fill w");
        view_w_                              = dense(dw_, w_dims_, types[2]);
        cu_seqlens_q_dims_                   = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_q_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_q_bytes = cu_seqlens_q_count * elementBytes(types[3]);
        host_.resize((cu_seqlens_q_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_q_count; ++j) { setHostValue(j, 0, types[3]); }
        for (std::size_t j = 0; j < cu_q.size(); ++j) { setHostValue(j, cu_q[j], types[3]); }
        checkHip(hipMalloc(&hcu_seqlens_q_, cu_seqlens_q_bytes), "malloc cu_seqlens_q");
        checkHip(hipMemcpy(hcu_seqlens_q_, host_.data(), cu_seqlens_q_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_q");
        view_cu_seqlens_q_  = dense(hcu_seqlens_q_, cu_seqlens_q_dims_, types[3]);
        cu_seqlens_kv_dims_ = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_kv_count =
            static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_kv_bytes = cu_seqlens_kv_count * elementBytes(types[4]);
        host_.resize((cu_seqlens_kv_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_kv_count; ++j) { setHostValue(j, 0, types[4]); }
        for (std::size_t j = 0; j < cu_k.size(); ++j) { setHostValue(j, cu_k[j], types[4]); }
        checkHip(hipMalloc(&hcu_seqlens_kv_, cu_seqlens_kv_bytes), "malloc cu_seqlens_kv");
        checkHip(hipMemcpy(hcu_seqlens_kv_, host_.data(), cu_seqlens_kv_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_kv");
        view_cu_seqlens_kv_                      = dense(hcu_seqlens_kv_, cu_seqlens_kv_dims_, types[4]);
        q_causal_offsets_dims_                   = {static_cast<std::int64_t>(segments.size())};
        const std::size_t q_causal_offsets_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size()));
        const std::size_t q_causal_offsets_bytes = q_causal_offsets_count * elementBytes(types[5]);
        host_.resize((q_causal_offsets_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_causal_offsets_count; ++j) { setHostValue(j, 0, types[5]); }
        checkHip(hipMalloc(&dq_causal_offsets_, q_causal_offsets_bytes), "malloc q_causal_offsets");
        checkHip(hipMemcpy(dq_causal_offsets_, host_.data(), q_causal_offsets_bytes, hipMemcpyHostToDevice),
                 "fill q_causal_offsets");
        view_q_causal_offsets_         = dense(dq_causal_offsets_, q_causal_offsets_dims_, types[5]);
        scores_dims_                   = {a, max_k};
        const std::size_t scores_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(max_k);
        const std::size_t scores_bytes = scores_count * elementBytes(types[6]);
        host_.resize((scores_bytes + 7) / 8);
        for (std::size_t j = 0; j < scores_count; ++j) { setHostValue(j, 123, types[6]); }
        checkHip(hipMalloc(&dscores_, scores_bytes), "malloc scores");
        checkHip(hipMemcpy(dscores_, host_.data(), scores_bytes, hipMemcpyHostToDevice), "fill scores");
        view_scores_          = dense(dscores_, scores_dims_, types[6]);
        ratio_value_          = static_cast<std::int64_t>(ratio);
        view_ratio_           = scalar(&ratio_value_, types[7]);
        sm_scale_value_       = static_cast<float>(0.25);
        view_sm_scale_        = scalar(&sm_scale_value_, types[8]);
        max_seqlen_kv_value_  = static_cast<std::int64_t>(max_k);
        view_max_seqlen_kv_   = scalar(&max_seqlen_kv_value_, types[9]);
        FTrainPattern pattern = nullptr;
        FTrainOps ops         = nullptr;
        FTrainArgs args       = nullptr;
        FTrainPlan plan       = nullptr;
        checkFtrain(ftrainPatternCreate(&pattern), "workspace pattern");
        buildPattern(pattern);
        checkFtrain(ftrainOpsCreate(&ops, pattern), "workspace ops");
        checkFtrain(ftrainArgsCreate(&args, ops), "workspace args");
        bindArgs(args);
        checkFtrain(ftrainPlanCreate(&plan, args), "workspace plan");
        checkFtrain(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace_bytes_), "workspace size");
        if (workspace_bytes_) { checkHip(hipMalloc(&workspace_, workspace_bytes_), "workspace allocation"); }
        ftrainPlanDestroy(plan);
        ftrainArgsDestroy(args);
        ftrainOpsDestroy(ops);
        ftrainPatternDestroy(pattern);
    }
    ~IndexerFwdThdSession() override {
        hipFree(dq_);
        hipFree(dk_);
        hipFree(dw_);
        hipFree(hcu_seqlens_q_);
        hipFree(hcu_seqlens_kv_);
        hipFree(dq_causal_offsets_);
        hipFree(dscores_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainIndexerFwdThd(view_q_, view_k_, view_w_, view_cu_seqlens_q_, view_cu_seqlens_kv_,
                                        view_q_causal_offsets_, view_scores_, view_ratio_, view_sm_scale_,
                                        view_max_seqlen_kv_, compute_, workspace_, workspace_bytes_, stream),
                    "ftrainIndexerFwdThd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_), "add q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tk_), "add k");
        checkFtrain(ftrainPatternAddTensor(pattern, &tw_), "add w");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_q_), "add cu_seqlens_q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_kv_), "add cu_seqlens_kv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_causal_offsets_), "add q_causal_offsets");
        checkFtrain(ftrainPatternAddTensor(pattern, &tscores_), "add scores");
        checkFtrain(ftrainPatternAddTensor(pattern, &tratio_), "add ratio");
        checkFtrain(ftrainPatternAddTensor(pattern, &tsm_scale_), "add sm_scale");
        checkFtrain(ftrainPatternAddTensor(pattern, &tmax_seqlen_kv_), "add max_seqlen_kv");
        checkFtrain(ftrainPatternAddIndexerFwdThd(pattern, &operation_, tq_, tk_, tw_, tcu_seqlens_q_, tcu_seqlens_kv_,
                                                  tq_causal_offsets_, tscores_, tratio_, tsm_scale_, tmax_seqlen_kv_),
                    "ftrainPatternAddIndexerFwdThd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tq_, view_q_), "set q");
        checkFtrain(ftrainArgsSetTensor(args, tk_, view_k_), "set k");
        checkFtrain(ftrainArgsSetTensor(args, tw_, view_w_), "set w");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_q_, view_cu_seqlens_q_), "set cu_seqlens_q");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_kv_, view_cu_seqlens_kv_), "set cu_seqlens_kv");
        checkFtrain(ftrainArgsSetTensor(args, tq_causal_offsets_, view_q_causal_offsets_), "set q_causal_offsets");
        checkFtrain(ftrainArgsSetTensor(args, tscores_, view_scores_), "set scores");
        checkFtrain(ftrainArgsSetTensor(args, tratio_, view_ratio_), "set ratio");
        checkFtrain(ftrainArgsSetTensor(args, tsm_scale_, view_sm_scale_), "set sm_scale");
        checkFtrain(ftrainArgsSetTensor(args, tmax_seqlen_kv_, view_max_seqlen_kv_), "set max_seqlen_kv");
        checkFtrain(ftrainArgsSetIndexerFwdThd(args, operation_, compute_), "ftrainArgsSetIndexerFwdThd");
    }

  private:
    static FTrainStorageView dense(void* memory, const std::vector<std::int64_t>& dims, FTrainNumericType type) {
        return {
            memory, dims.data(), nullptr, static_cast<std::uint8_t>(dims.size()), type, FTRAIN_INDEX_TYPE_CONTINUOUS,
            false};
    }
    static FTrainStorageView scalar(void* value, FTrainNumericType type) {
        return {value, nullptr, nullptr, 0, type, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    }
    static std::size_t elementBytes(FTrainNumericType type) {
        return (type == FTRAIN_NUMERIC_TYPE_INT64)
                   ? 8
                   : ((type == FTRAIN_NUMERIC_TYPE_BF16 || type == FTRAIN_NUMERIC_TYPE_FP16) ? 2 : (4));
    }
    void setHostValue(std::size_t index, double value, FTrainNumericType type) {
        auto* dst = reinterpret_cast<unsigned char*>(host_.data()) + index * elementBytes(type);
        if (type == FTRAIN_NUMERIC_TYPE_FP32) {
            const float v = static_cast<float>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_FP16) {
            const __half v = __float2half(static_cast<float>(value));
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_BF16) {
            const float v = static_cast<float>(value);
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            bits                        += 0x7fff + ((bits >> 16) & 1);
            const std::uint16_t rounded  = static_cast<std::uint16_t>(bits >> 16);
            std::memcpy(dst, &rounded, sizeof(rounded));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT64) {
            const std::int64_t v = static_cast<std::int64_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT32) {
            const std::int32_t v = static_cast<std::int32_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> q_dims_;
    void* dq_ = nullptr;
    std::vector<std::int64_t> k_dims_;
    void* dk_ = nullptr;
    std::vector<std::int64_t> w_dims_;
    void* dw_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_q_dims_;
    void* hcu_seqlens_q_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_kv_dims_;
    void* hcu_seqlens_kv_ = nullptr;
    std::vector<std::int64_t> q_causal_offsets_dims_;
    void* dq_causal_offsets_ = nullptr;
    std::vector<std::int64_t> scores_dims_;
    void* dscores_ = nullptr;
    std::int64_t ratio_value_{};
    float sm_scale_value_{};
    std::int64_t max_seqlen_kv_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_q_{};
    FTrainTensorId tq_{};
    FTrainStorageView view_k_{};
    FTrainTensorId tk_{};
    FTrainStorageView view_w_{};
    FTrainTensorId tw_{};
    FTrainStorageView view_cu_seqlens_q_{};
    FTrainTensorId tcu_seqlens_q_{};
    FTrainStorageView view_cu_seqlens_kv_{};
    FTrainTensorId tcu_seqlens_kv_{};
    FTrainStorageView view_q_causal_offsets_{};
    FTrainTensorId tq_causal_offsets_{};
    FTrainStorageView view_scores_{};
    FTrainTensorId tscores_{};
    FTrainStorageView view_ratio_{};
    FTrainTensorId tratio_{};
    FTrainStorageView view_sm_scale_{};
    FTrainTensorId tsm_scale_{};
    FTrainStorageView view_max_seqlen_kv_{};
    FTrainTensorId tmax_seqlen_kv_{};
    FTrainIndexerFwdThdOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct IndexerFwdThdCase final : OpCase {
    const char* name() const override { return "indexer_fwd_thd"; }
    const char* shapeGrammar() const override { return "tokens,heads,head_dim"; }
    const char* convenienceImpl() const override { return "ftrainIndexerFwdThd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"bf16", FTRAIN_NUMERIC_TYPE_BF16},
            {"fp16", FTRAIN_NUMERIC_TYPE_FP16},
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "tokens,heads,head_dim: three positive values required");
        return std::make_unique<IndexerFwdThdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<IndexerFwdThdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
