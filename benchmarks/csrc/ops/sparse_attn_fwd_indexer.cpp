// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
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

struct SparseAttnFwdIndexerSession final : OpSession {
    SparseAttnFwdIndexerSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t indexer_topk = 32;
        std::int64_t batch = 1, kv_tokens = a, top_k = 128, ratio = 1;
        if (shape[0] == 4096 && shape[1] == 64 && shape[2] == 512) {
            kv_tokens    = 5120;
            top_k        = 640;
            indexer_topk = 512;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 512) {
            kv_tokens    = 5120;
            top_k        = 1152;
            indexer_topk = 1024;
        }
        if (shape[0] == 8192 && shape[1] == 8 && shape[2] == 512) {
            kv_tokens    = 10240;
            top_k        = 640;
            indexer_topk = 512;
        }
        if (shape[0] == 8192 && shape[1] == 16 && shape[2] == 512) {
            kv_tokens    = 10240;
            top_k        = 1152;
            indexer_topk = 1024;
        }
        if (shape[0] == 4096 && shape[1] == 64 && shape[2] == 576) {
            kv_tokens    = 4096;
            top_k        = 2048;
            indexer_topk = 2048;
        }
        if (shape[0] == 8192 && shape[1] == 8 && shape[2] == 576) {
            kv_tokens    = 8192;
            top_k        = 2048;
            indexer_topk = 2048;
        }
        std::array<FTrainNumericType, 11> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            default: require(false, "unsupported precision");
        }
        q_dims_ = {a, b, c};
        const std::size_t q_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t q_bytes = q_count * elementBytes(types[0]);
        host_.resize((q_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_count; ++j) { setHostValue(j, 0, types[0]); }
        checkHip(hipMalloc(&dq_, q_bytes), "malloc q");
        checkHip(hipMemcpy(dq_, host_.data(), q_bytes, hipMemcpyHostToDevice), "fill q");
        view_q_                    = dense(dq_, q_dims_, types[0]);
        kv_dims_                   = {kv_tokens, c};
        const std::size_t kv_count = static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(c);
        const std::size_t kv_bytes = kv_count * elementBytes(types[1]);
        host_.resize((kv_bytes + 7) / 8);
        for (std::size_t j = 0; j < kv_count; ++j) { setHostValue(j, 0.5, types[1]); }
        checkHip(hipMalloc(&dkv_, kv_bytes), "malloc kv");
        checkHip(hipMemcpy(dkv_, host_.data(), kv_bytes, hipMemcpyHostToDevice), "fill kv");
        view_kv_                          = dense(dkv_, kv_dims_, types[1]);
        attn_sink_dims_                   = {b};
        const std::size_t attn_sink_count = static_cast<std::size_t>(b);
        const std::size_t attn_sink_bytes = attn_sink_count * elementBytes(types[2]);
        host_.resize((attn_sink_bytes + 7) / 8);
        for (std::size_t j = 0; j < attn_sink_count; ++j) { setHostValue(j, 0, types[2]); }
        checkHip(hipMalloc(&dattn_sink_, attn_sink_bytes), "malloc attn_sink");
        checkHip(hipMemcpy(dattn_sink_, host_.data(), attn_sink_bytes, hipMemcpyHostToDevice), "fill attn_sink");
        view_attn_sink_                      = dense(dattn_sink_, attn_sink_dims_, types[2]);
        topk_indices_dims_                   = {a, top_k};
        const std::size_t topk_indices_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(top_k);
        const std::size_t topk_indices_bytes = topk_indices_count * elementBytes(types[3]);
        host_.resize((topk_indices_bytes + 7) / 8);
        for (std::size_t j = 0; j < topk_indices_count; ++j) { setHostValue(j, (j % top_k) % kv_tokens, types[3]); }
        checkHip(hipMalloc(&dtopk_indices_, topk_indices_bytes), "malloc topk_indices");
        checkHip(hipMemcpy(dtopk_indices_, host_.data(), topk_indices_bytes, hipMemcpyHostToDevice),
                 "fill topk_indices");
        view_topk_indices_                  = dense(dtopk_indices_, topk_indices_dims_, types[3]);
        topk_length_dims_                   = {a};
        const std::size_t topk_length_count = static_cast<std::size_t>(a);
        const std::size_t topk_length_bytes = topk_length_count * elementBytes(types[4]);
        host_.resize((topk_length_bytes + 7) / 8);
        for (std::size_t j = 0; j < topk_length_count; ++j) { setHostValue(j, top_k, types[4]); }
        checkHip(hipMalloc(&dtopk_length_, topk_length_bytes), "malloc topk_length");
        checkHip(hipMemcpy(dtopk_length_, host_.data(), topk_length_bytes, hipMemcpyHostToDevice), "fill topk_length");
        view_topk_length_ = dense(dtopk_length_, topk_length_dims_, types[4]);
        out_dims_         = {a, b, 512};
        const std::size_t out_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(512);
        const std::size_t out_bytes = out_count * elementBytes(types[5]);
        host_.resize((out_bytes + 7) / 8);
        for (std::size_t j = 0; j < out_count; ++j) { setHostValue(j, 0.5 * top_k / (top_k + 1.0), types[5]); }
        checkHip(hipMalloc(&dout_, out_bytes), "malloc out");
        checkHip(hipMemcpy(dout_, host_.data(), out_bytes, hipMemcpyHostToDevice), "fill out");
        view_out_                          = dense(dout_, out_dims_, types[5]);
        max_logits_dims_                   = {a, b};
        const std::size_t max_logits_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t max_logits_bytes = max_logits_count * elementBytes(types[6]);
        host_.resize((max_logits_bytes + 7) / 8);
        for (std::size_t j = 0; j < max_logits_count; ++j) { setHostValue(j, 123, types[6]); }
        checkHip(hipMalloc(&dmax_logits_, max_logits_bytes), "malloc max_logits");
        checkHip(hipMemcpy(dmax_logits_, host_.data(), max_logits_bytes, hipMemcpyHostToDevice), "fill max_logits");
        view_max_logits_            = dense(dmax_logits_, max_logits_dims_, types[6]);
        lse_dims_                   = {a, b};
        const std::size_t lse_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t lse_bytes = lse_count * elementBytes(types[7]);
        host_.resize((lse_bytes + 7) / 8);
        for (std::size_t j = 0; j < lse_count; ++j) { setHostValue(j, std::log(static_cast<double>(top_k)), types[7]); }
        checkHip(hipMalloc(&dlse_, lse_bytes), "malloc lse");
        checkHip(hipMemcpy(dlse_, host_.data(), lse_bytes, hipMemcpyHostToDevice), "fill lse");
        view_lse_                           = dense(dlse_, lse_dims_, types[7]);
        lse_indexer_dims_                   = {a, b};
        const std::size_t lse_indexer_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t lse_indexer_bytes = lse_indexer_count * elementBytes(types[8]);
        host_.resize((lse_indexer_bytes + 7) / 8);
        for (std::size_t j = 0; j < lse_indexer_count; ++j) { setHostValue(j, 123, types[8]); }
        checkHip(hipMalloc(&dlse_indexer_, lse_indexer_bytes), "malloc lse_indexer");
        checkHip(hipMemcpy(dlse_indexer_, host_.data(), lse_indexer_bytes, hipMemcpyHostToDevice), "fill lse_indexer");
        view_lse_indexer_     = dense(dlse_indexer_, lse_indexer_dims_, types[8]);
        softmax_scale_value_  = static_cast<float>(0.125);
        view_softmax_scale_   = scalar(&softmax_scale_value_, types[9]);
        indexer_topk_value_   = static_cast<std::int64_t>(indexer_topk);
        view_indexer_topk_    = scalar(&indexer_topk_value_, types[10]);
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
    ~SparseAttnFwdIndexerSession() override {
        hipFree(dq_);
        hipFree(dkv_);
        hipFree(dattn_sink_);
        hipFree(dtopk_indices_);
        hipFree(dtopk_length_);
        hipFree(dout_);
        hipFree(dmax_logits_);
        hipFree(dlse_);
        hipFree(dlse_indexer_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(
            ftrainSparseAttnFwdIndexer(view_q_, view_kv_, view_attn_sink_, view_topk_indices_, view_topk_length_,
                                       view_out_, view_max_logits_, view_lse_, view_lse_indexer_, view_softmax_scale_,
                                       view_indexer_topk_, compute_, workspace_, workspace_bytes_, stream),
            "ftrainSparseAttnFwdIndexer");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_), "add q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tkv_), "add kv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tattn_sink_), "add attn_sink");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_indices_), "add topk_indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_length_), "add topk_length");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_), "add out");
        checkFtrain(ftrainPatternAddTensor(pattern, &tmax_logits_), "add max_logits");
        checkFtrain(ftrainPatternAddTensor(pattern, &tlse_), "add lse");
        checkFtrain(ftrainPatternAddTensor(pattern, &tlse_indexer_), "add lse_indexer");
        checkFtrain(ftrainPatternAddTensor(pattern, &tsoftmax_scale_), "add softmax_scale");
        checkFtrain(ftrainPatternAddTensor(pattern, &tindexer_topk_), "add indexer_topk");
        checkFtrain(ftrainPatternAddSparseAttnFwdIndexer(pattern, &operation_, tq_, tkv_, tattn_sink_, ttopk_indices_,
                                                         ttopk_length_, tout_, tmax_logits_, tlse_, tlse_indexer_,
                                                         tsoftmax_scale_, tindexer_topk_),
                    "ftrainPatternAddSparseAttnFwdIndexer");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tq_, view_q_), "set q");
        checkFtrain(ftrainArgsSetTensor(args, tkv_, view_kv_), "set kv");
        checkFtrain(ftrainArgsSetTensor(args, tattn_sink_, view_attn_sink_), "set attn_sink");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_indices_, view_topk_indices_), "set topk_indices");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_length_, view_topk_length_), "set topk_length");
        checkFtrain(ftrainArgsSetTensor(args, tout_, view_out_), "set out");
        checkFtrain(ftrainArgsSetTensor(args, tmax_logits_, view_max_logits_), "set max_logits");
        checkFtrain(ftrainArgsSetTensor(args, tlse_, view_lse_), "set lse");
        checkFtrain(ftrainArgsSetTensor(args, tlse_indexer_, view_lse_indexer_), "set lse_indexer");
        checkFtrain(ftrainArgsSetTensor(args, tsoftmax_scale_, view_softmax_scale_), "set softmax_scale");
        checkFtrain(ftrainArgsSetTensor(args, tindexer_topk_, view_indexer_topk_), "set indexer_topk");
        checkFtrain(ftrainArgsSetSparseAttnFwdIndexer(args, operation_, compute_), "ftrainArgsSetSparseAttnFwdIndexer");
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
    std::vector<std::int64_t> kv_dims_;
    void* dkv_ = nullptr;
    std::vector<std::int64_t> attn_sink_dims_;
    void* dattn_sink_ = nullptr;
    std::vector<std::int64_t> topk_indices_dims_;
    void* dtopk_indices_ = nullptr;
    std::vector<std::int64_t> topk_length_dims_;
    void* dtopk_length_ = nullptr;
    std::vector<std::int64_t> out_dims_;
    void* dout_ = nullptr;
    std::vector<std::int64_t> max_logits_dims_;
    void* dmax_logits_ = nullptr;
    std::vector<std::int64_t> lse_dims_;
    void* dlse_ = nullptr;
    std::vector<std::int64_t> lse_indexer_dims_;
    void* dlse_indexer_ = nullptr;
    float softmax_scale_value_{};
    std::int64_t indexer_topk_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_q_{};
    FTrainTensorId tq_{};
    FTrainStorageView view_kv_{};
    FTrainTensorId tkv_{};
    FTrainStorageView view_attn_sink_{};
    FTrainTensorId tattn_sink_{};
    FTrainStorageView view_topk_indices_{};
    FTrainTensorId ttopk_indices_{};
    FTrainStorageView view_topk_length_{};
    FTrainTensorId ttopk_length_{};
    FTrainStorageView view_out_{};
    FTrainTensorId tout_{};
    FTrainStorageView view_max_logits_{};
    FTrainTensorId tmax_logits_{};
    FTrainStorageView view_lse_{};
    FTrainTensorId tlse_{};
    FTrainStorageView view_lse_indexer_{};
    FTrainTensorId tlse_indexer_{};
    FTrainStorageView view_softmax_scale_{};
    FTrainTensorId tsoftmax_scale_{};
    FTrainStorageView view_indexer_topk_{};
    FTrainTensorId tindexer_topk_{};
    FTrainSparseAttnFwdIndexerOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct SparseAttnFwdIndexerCase final : OpCase {
    const char* name() const override { return "sparse_attn_fwd_indexer"; }
    const char* shapeGrammar() const override { return "tokens,heads,head_dim"; }
    const char* convenienceImpl() const override { return "ftrainSparseAttnFwdIndexer"; }
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
        return std::make_unique<SparseAttnFwdIndexerSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<SparseAttnFwdIndexerCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
