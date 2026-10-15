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

struct SparseAttnBwdSession final : OpSession {
    SparseAttnBwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t batch = 1, kv_tokens = a, top_k = 128, ratio = 1;
        if (shape[0] == 4096 && shape[1] == 64 && shape[2] == 512) {
            kv_tokens = 5120;
            top_k     = 640;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 512) {
            kv_tokens = 5120;
            top_k     = 1152;
        }
        if (shape[0] == 8192 && shape[1] == 8 && shape[2] == 512) {
            kv_tokens = 10240;
            top_k     = 640;
        }
        if (shape[0] == 8192 && shape[1] == 16 && shape[2] == 512) {
            kv_tokens = 10240;
            top_k     = 1152;
        }
        if (shape[0] == 16384 && shape[1] == 8 && shape[2] == 512) {
            kv_tokens = 16512;
            top_k     = 256;
        }
        if (shape[0] == 4096 && shape[1] == 64 && shape[2] == 576) {
            kv_tokens = 4096;
            top_k     = 2048;
        }
        if (shape[0] == 4096 && shape[1] == 8 && shape[2] == 576) {
            kv_tokens = 4096;
            top_k     = 2048;
        }
        std::array<FTrainNumericType, 12> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32};
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
        view_kv_  = dense(dkv_, kv_dims_, types[1]);
        out_dims_ = {a, b, 512};
        const std::size_t out_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(512);
        const std::size_t out_bytes = out_count * elementBytes(types[2]);
        host_.resize((out_bytes + 7) / 8);
        for (std::size_t j = 0; j < out_count; ++j) { setHostValue(j, 0.5 * top_k / (top_k + 1.0), types[2]); }
        checkHip(hipMalloc(&dout_, out_bytes), "malloc out");
        checkHip(hipMemcpy(dout_, host_.data(), out_bytes, hipMemcpyHostToDevice), "fill out");
        view_out_  = dense(dout_, out_dims_, types[2]);
        dout_dims_ = {a, b, 512};
        const std::size_t dout_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(512);
        const std::size_t dout_bytes = dout_count * elementBytes(types[3]);
        host_.resize((dout_bytes + 7) / 8);
        for (std::size_t j = 0; j < dout_count; ++j) { setHostValue(j, 0, types[3]); }
        checkHip(hipMalloc(&ddout_, dout_bytes), "malloc dout");
        checkHip(hipMemcpy(ddout_, host_.data(), dout_bytes, hipMemcpyHostToDevice), "fill dout");
        view_dout_                  = dense(ddout_, dout_dims_, types[3]);
        lse_dims_                   = {a, b};
        const std::size_t lse_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t lse_bytes = lse_count * elementBytes(types[4]);
        host_.resize((lse_bytes + 7) / 8);
        for (std::size_t j = 0; j < lse_count; ++j) { setHostValue(j, std::log(static_cast<double>(top_k)), types[4]); }
        checkHip(hipMalloc(&dlse_, lse_bytes), "malloc lse");
        checkHip(hipMemcpy(dlse_, host_.data(), lse_bytes, hipMemcpyHostToDevice), "fill lse");
        view_lse_                         = dense(dlse_, lse_dims_, types[4]);
        attn_sink_dims_                   = {b};
        const std::size_t attn_sink_count = static_cast<std::size_t>(b);
        const std::size_t attn_sink_bytes = attn_sink_count * elementBytes(types[5]);
        host_.resize((attn_sink_bytes + 7) / 8);
        for (std::size_t j = 0; j < attn_sink_count; ++j) { setHostValue(j, 0, types[5]); }
        checkHip(hipMalloc(&dattn_sink_, attn_sink_bytes), "malloc attn_sink");
        checkHip(hipMemcpy(dattn_sink_, host_.data(), attn_sink_bytes, hipMemcpyHostToDevice), "fill attn_sink");
        view_attn_sink_                      = dense(dattn_sink_, attn_sink_dims_, types[5]);
        topk_indices_dims_                   = {a, top_k};
        const std::size_t topk_indices_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(top_k);
        const std::size_t topk_indices_bytes = topk_indices_count * elementBytes(types[6]);
        host_.resize((topk_indices_bytes + 7) / 8);
        for (std::size_t j = 0; j < topk_indices_count; ++j) { setHostValue(j, (j % top_k) % kv_tokens, types[6]); }
        checkHip(hipMalloc(&dtopk_indices_, topk_indices_bytes), "malloc topk_indices");
        checkHip(hipMemcpy(dtopk_indices_, host_.data(), topk_indices_bytes, hipMemcpyHostToDevice),
                 "fill topk_indices");
        view_topk_indices_                  = dense(dtopk_indices_, topk_indices_dims_, types[6]);
        topk_length_dims_                   = {a};
        const std::size_t topk_length_count = static_cast<std::size_t>(a);
        const std::size_t topk_length_bytes = topk_length_count * elementBytes(types[7]);
        host_.resize((topk_length_bytes + 7) / 8);
        for (std::size_t j = 0; j < topk_length_count; ++j) { setHostValue(j, top_k, types[7]); }
        checkHip(hipMalloc(&dtopk_length_, topk_length_bytes), "malloc topk_length");
        checkHip(hipMemcpy(dtopk_length_, host_.data(), topk_length_bytes, hipMemcpyHostToDevice), "fill topk_length");
        view_topk_length_ = dense(dtopk_length_, topk_length_dims_, types[7]);
        dq_dims_          = {a, b, c};
        const std::size_t dq_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t dq_bytes = dq_count * elementBytes(types[8]);
        host_.resize((dq_bytes + 7) / 8);
        for (std::size_t j = 0; j < dq_count; ++j) { setHostValue(j, 123, types[8]); }
        checkHip(hipMalloc(&ddq_, dq_bytes), "malloc dq");
        checkHip(hipMemcpy(ddq_, host_.data(), dq_bytes, hipMemcpyHostToDevice), "fill dq");
        view_dq_                    = dense(ddq_, dq_dims_, types[8]);
        dkv_dims_                   = {kv_tokens, c};
        const std::size_t dkv_count = static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(c);
        const std::size_t dkv_bytes = dkv_count * elementBytes(types[9]);
        host_.resize((dkv_bytes + 7) / 8);
        for (std::size_t j = 0; j < dkv_count; ++j) { setHostValue(j, 123, types[9]); }
        checkHip(hipMalloc(&ddkv_, dkv_bytes), "malloc dkv");
        checkHip(hipMemcpy(ddkv_, host_.data(), dkv_bytes, hipMemcpyHostToDevice), "fill dkv");
        view_dkv_                     = dense(ddkv_, dkv_dims_, types[9]);
        dsink_dims_                   = {b};
        const std::size_t dsink_count = static_cast<std::size_t>(b);
        const std::size_t dsink_bytes = dsink_count * elementBytes(types[10]);
        host_.resize((dsink_bytes + 7) / 8);
        for (std::size_t j = 0; j < dsink_count; ++j) { setHostValue(j, 123, types[10]); }
        checkHip(hipMalloc(&ddsink_, dsink_bytes), "malloc dsink");
        checkHip(hipMemcpy(ddsink_, host_.data(), dsink_bytes, hipMemcpyHostToDevice), "fill dsink");
        view_dsink_           = dense(ddsink_, dsink_dims_, types[10]);
        softmax_scale_value_  = static_cast<float>(0.125);
        view_softmax_scale_   = scalar(&softmax_scale_value_, types[11]);
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
    ~SparseAttnBwdSession() override {
        hipFree(dq_);
        hipFree(dkv_);
        hipFree(dout_);
        hipFree(ddout_);
        hipFree(dlse_);
        hipFree(dattn_sink_);
        hipFree(dtopk_indices_);
        hipFree(dtopk_length_);
        hipFree(ddq_);
        hipFree(ddkv_);
        hipFree(ddsink_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainSparseAttnBwd(view_q_, view_kv_, view_out_, view_dout_, view_lse_, view_attn_sink_,
                                        view_topk_indices_, view_topk_length_, view_dq_, view_dkv_, view_dsink_,
                                        view_softmax_scale_, compute_, workspace_, workspace_bytes_, stream),
                    "ftrainSparseAttnBwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_), "add q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tkv_), "add kv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_), "add out");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdout_), "add dout");
        checkFtrain(ftrainPatternAddTensor(pattern, &tlse_), "add lse");
        checkFtrain(ftrainPatternAddTensor(pattern, &tattn_sink_), "add attn_sink");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_indices_), "add topk_indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_length_), "add topk_length");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdq_), "add dq");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdkv_), "add dkv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdsink_), "add dsink");
        checkFtrain(ftrainPatternAddTensor(pattern, &tsoftmax_scale_), "add softmax_scale");
        checkFtrain(ftrainPatternAddSparseAttnBwd(pattern, &operation_, tq_, tkv_, tout_, tdout_, tlse_, tattn_sink_,
                                                  ttopk_indices_, ttopk_length_, tdq_, tdkv_, tdsink_, tsoftmax_scale_),
                    "ftrainPatternAddSparseAttnBwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tq_, view_q_), "set q");
        checkFtrain(ftrainArgsSetTensor(args, tkv_, view_kv_), "set kv");
        checkFtrain(ftrainArgsSetTensor(args, tout_, view_out_), "set out");
        checkFtrain(ftrainArgsSetTensor(args, tdout_, view_dout_), "set dout");
        checkFtrain(ftrainArgsSetTensor(args, tlse_, view_lse_), "set lse");
        checkFtrain(ftrainArgsSetTensor(args, tattn_sink_, view_attn_sink_), "set attn_sink");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_indices_, view_topk_indices_), "set topk_indices");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_length_, view_topk_length_), "set topk_length");
        checkFtrain(ftrainArgsSetTensor(args, tdq_, view_dq_), "set dq");
        checkFtrain(ftrainArgsSetTensor(args, tdkv_, view_dkv_), "set dkv");
        checkFtrain(ftrainArgsSetTensor(args, tdsink_, view_dsink_), "set dsink");
        checkFtrain(ftrainArgsSetTensor(args, tsoftmax_scale_, view_softmax_scale_), "set softmax_scale");
        checkFtrain(ftrainArgsSetSparseAttnBwd(args, operation_, compute_), "ftrainArgsSetSparseAttnBwd");
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
        return (type == FTRAIN_NUMERIC_TYPE_BF16 || type == FTRAIN_NUMERIC_TYPE_FP16) ? 2 : (4);
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
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> q_dims_;
    void* dq_ = nullptr;
    std::vector<std::int64_t> kv_dims_;
    void* dkv_ = nullptr;
    std::vector<std::int64_t> out_dims_;
    void* dout_ = nullptr;
    std::vector<std::int64_t> dout_dims_;
    void* ddout_ = nullptr;
    std::vector<std::int64_t> lse_dims_;
    void* dlse_ = nullptr;
    std::vector<std::int64_t> attn_sink_dims_;
    void* dattn_sink_ = nullptr;
    std::vector<std::int64_t> topk_indices_dims_;
    void* dtopk_indices_ = nullptr;
    std::vector<std::int64_t> topk_length_dims_;
    void* dtopk_length_ = nullptr;
    std::vector<std::int64_t> dq_dims_;
    void* ddq_ = nullptr;
    std::vector<std::int64_t> dkv_dims_;
    void* ddkv_ = nullptr;
    std::vector<std::int64_t> dsink_dims_;
    void* ddsink_ = nullptr;
    float softmax_scale_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_q_{};
    FTrainTensorId tq_{};
    FTrainStorageView view_kv_{};
    FTrainTensorId tkv_{};
    FTrainStorageView view_out_{};
    FTrainTensorId tout_{};
    FTrainStorageView view_dout_{};
    FTrainTensorId tdout_{};
    FTrainStorageView view_lse_{};
    FTrainTensorId tlse_{};
    FTrainStorageView view_attn_sink_{};
    FTrainTensorId tattn_sink_{};
    FTrainStorageView view_topk_indices_{};
    FTrainTensorId ttopk_indices_{};
    FTrainStorageView view_topk_length_{};
    FTrainTensorId ttopk_length_{};
    FTrainStorageView view_dq_{};
    FTrainTensorId tdq_{};
    FTrainStorageView view_dkv_{};
    FTrainTensorId tdkv_{};
    FTrainStorageView view_dsink_{};
    FTrainTensorId tdsink_{};
    FTrainStorageView view_softmax_scale_{};
    FTrainTensorId tsoftmax_scale_{};
    FTrainSparseAttnBwdOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct SparseAttnBwdCase final : OpCase {
    const char* name() const override { return "sparse_attn_bwd"; }
    const char* shapeGrammar() const override { return "tokens,heads,head_dim"; }
    const char* convenienceImpl() const override { return "ftrainSparseAttnBwd"; }
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
        return std::make_unique<SparseAttnBwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<SparseAttnBwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
