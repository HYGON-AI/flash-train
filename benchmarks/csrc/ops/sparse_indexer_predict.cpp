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

struct SparseIndexerPredictSession final : OpSession {
    SparseIndexerPredictSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        b                  = 64;
        c                  = shape[1];
        std::int64_t batch = 1, kv_tokens = std::max<std::int64_t>(a / 4, 1), top_k = shape[2], ratio = 1;
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 512) {
            batch     = 1;
            kv_tokens = 1024;
            ratio     = 4;
            b         = 64;
            top_k     = 512;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 1024) {
            batch     = 1;
            kv_tokens = 1024;
            ratio     = 4;
            b         = 64;
            top_k     = 1024;
        }
        if (shape[0] == 8192 && shape[1] == 128 && shape[2] == 512) {
            batch     = 1;
            kv_tokens = 2048;
            ratio     = 4;
            b         = 8;
            top_k     = 512;
        }
        if (shape[0] == 16384 && shape[1] == 128 && shape[2] == 128) {
            batch     = 1;
            kv_tokens = 128;
            ratio     = 128;
            b         = 64;
            top_k     = 128;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 2048) {
            batch     = 1;
            kv_tokens = 4096;
            ratio     = 1;
            b         = 32;
            top_k     = 2048;
        }
        if (shape[0] == 8192 && shape[1] == 128 && shape[2] == 2048) {
            batch     = 1;
            kv_tokens = 8192;
            ratio     = 1;
            b         = 4;
            top_k     = 2048;
        }
        std::array<FTrainNumericType, 7> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_UINT8};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_UINT8};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_UINT8};
                break;
            default: require(false, "unsupported precision");
        }
        q_dims_                   = {batch, a, b, c};
        const std::size_t q_count = static_cast<std::size_t>(batch) * static_cast<std::size_t>(a) *
                                    static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t q_bytes = q_count * elementBytes(types[0]);
        host_.resize((q_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_count; ++j) { setHostValue(j, 0.25, types[0]); }
        checkHip(hipMalloc(&dq_, q_bytes), "malloc q");
        checkHip(hipMemcpy(dq_, host_.data(), q_bytes, hipMemcpyHostToDevice), "fill q");
        view_q_ = dense(dq_, q_dims_, types[0]);
        k_dims_ = {batch, kv_tokens, c};
        const std::size_t k_count =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(c);
        const std::size_t k_bytes = k_count * elementBytes(types[1]);
        host_.resize((k_bytes + 7) / 8);
        for (std::size_t j = 0; j < k_count; ++j) { setHostValue(j, 0.5, types[1]); }
        checkHip(hipMalloc(&dk_, k_bytes), "malloc k");
        checkHip(hipMemcpy(dk_, host_.data(), k_bytes, hipMemcpyHostToDevice), "fill k");
        view_k_ = dense(dk_, k_dims_, types[1]);
        w_dims_ = {batch, a, b};
        const std::size_t w_count =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t w_bytes = w_count * elementBytes(types[2]);
        host_.resize((w_bytes + 7) / 8);
        for (std::size_t j = 0; j < w_count; ++j) { setHostValue(j, 1, types[2]); }
        checkHip(hipMalloc(&dw_, w_bytes), "malloc w");
        checkHip(hipMemcpy(dw_, host_.data(), w_bytes, hipMemcpyHostToDevice), "fill w");
        view_w_            = dense(dw_, w_dims_, types[2]);
        topk_indices_dims_ = {batch, a, top_k};
        const std::size_t topk_indices_count =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(a) * static_cast<std::size_t>(top_k);
        const std::size_t topk_indices_bytes = topk_indices_count * elementBytes(types[3]);
        host_.resize((topk_indices_bytes + 7) / 8);
        for (std::size_t j = 0; j < topk_indices_count; ++j) { setHostValue(j, (j % top_k) % kv_tokens, types[3]); }
        checkHip(hipMalloc(&dtopk_indices_, topk_indices_bytes), "malloc topk_indices");
        checkHip(hipMemcpy(dtopk_indices_, host_.data(), topk_indices_bytes, hipMemcpyHostToDevice),
                 "fill topk_indices");
        view_topk_indices_ = dense(dtopk_indices_, topk_indices_dims_, types[3]);
        predict_dims_      = {batch, a, top_k};
        const std::size_t predict_count =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(a) * static_cast<std::size_t>(top_k);
        const std::size_t predict_bytes = predict_count * elementBytes(types[4]);
        host_.resize((predict_bytes + 7) / 8);
        for (std::size_t j = 0; j < predict_count; ++j) { setHostValue(j, 123, types[4]); }
        checkHip(hipMalloc(&dpredict_, predict_bytes), "malloc predict");
        checkHip(hipMemcpy(dpredict_, host_.data(), predict_bytes, hipMemcpyHostToDevice), "fill predict");
        view_predict_              = dense(dpredict_, predict_dims_, types[4]);
        sm_scale_value_            = static_cast<float>(0.25);
        view_sm_scale_             = scalar(&sm_scale_value_, types[5]);
        topk_indices_global_value_ = static_cast<std::uint8_t>(0);
        view_topk_indices_global_  = scalar(&topk_indices_global_value_, types[6]);
        FTrainPattern pattern      = nullptr;
        FTrainOps ops              = nullptr;
        FTrainArgs args            = nullptr;
        FTrainPlan plan            = nullptr;
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
    ~SparseIndexerPredictSession() override {
        hipFree(dq_);
        hipFree(dk_);
        hipFree(dw_);
        hipFree(dtopk_indices_);
        hipFree(dpredict_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(
            ftrainSparseIndexerPredict(view_q_, view_k_, view_w_, view_topk_indices_, view_predict_, view_sm_scale_,
                                       view_topk_indices_global_, compute_, workspace_, workspace_bytes_, stream),
            "ftrainSparseIndexerPredict");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_), "add q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tk_), "add k");
        checkFtrain(ftrainPatternAddTensor(pattern, &tw_), "add w");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_indices_), "add topk_indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &tpredict_), "add predict");
        checkFtrain(ftrainPatternAddTensor(pattern, &tsm_scale_), "add sm_scale");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_indices_global_), "add topk_indices_global");
        checkFtrain(ftrainPatternAddSparseIndexerPredict(pattern, &operation_, tq_, tk_, tw_, ttopk_indices_, tpredict_,
                                                         tsm_scale_, ttopk_indices_global_),
                    "ftrainPatternAddSparseIndexerPredict");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tq_, view_q_), "set q");
        checkFtrain(ftrainArgsSetTensor(args, tk_, view_k_), "set k");
        checkFtrain(ftrainArgsSetTensor(args, tw_, view_w_), "set w");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_indices_, view_topk_indices_), "set topk_indices");
        checkFtrain(ftrainArgsSetTensor(args, tpredict_, view_predict_), "set predict");
        checkFtrain(ftrainArgsSetTensor(args, tsm_scale_, view_sm_scale_), "set sm_scale");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_indices_global_, view_topk_indices_global_),
                    "set topk_indices_global");
        checkFtrain(ftrainArgsSetSparseIndexerPredict(args, operation_, compute_), "ftrainArgsSetSparseIndexerPredict");
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
        return (type == FTRAIN_NUMERIC_TYPE_BF16 || type == FTRAIN_NUMERIC_TYPE_FP16)
                   ? 2
                   : ((type == FTRAIN_NUMERIC_TYPE_UINT8) ? 1 : (4));
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
        } else if (type == FTRAIN_NUMERIC_TYPE_INT32) {
            const std::int32_t v = static_cast<std::int32_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_UINT8) {
            *dst = static_cast<std::uint8_t>(value);
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
    std::vector<std::int64_t> topk_indices_dims_;
    void* dtopk_indices_ = nullptr;
    std::vector<std::int64_t> predict_dims_;
    void* dpredict_ = nullptr;
    float sm_scale_value_{};
    std::uint8_t topk_indices_global_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_q_{};
    FTrainTensorId tq_{};
    FTrainStorageView view_k_{};
    FTrainTensorId tk_{};
    FTrainStorageView view_w_{};
    FTrainTensorId tw_{};
    FTrainStorageView view_topk_indices_{};
    FTrainTensorId ttopk_indices_{};
    FTrainStorageView view_predict_{};
    FTrainTensorId tpredict_{};
    FTrainStorageView view_sm_scale_{};
    FTrainTensorId tsm_scale_{};
    FTrainStorageView view_topk_indices_global_{};
    FTrainTensorId ttopk_indices_global_{};
    FTrainSparseIndexerPredictOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct SparseIndexerPredictCase final : OpCase {
    const char* name() const override { return "sparse_indexer_predict"; }
    const char* shapeGrammar() const override { return "query_tokens,head_dim,top_k"; }
    const char* convenienceImpl() const override { return "ftrainSparseIndexerPredict"; }
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
                "query_tokens,head_dim,top_k: three positive values required");
        return std::make_unique<SparseIndexerPredictSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<SparseIndexerPredictCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
