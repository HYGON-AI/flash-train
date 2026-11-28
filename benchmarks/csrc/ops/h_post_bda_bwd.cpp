// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

struct HPostBdaBwdSession final : OpSession {
    HPostBdaBwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 11> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            default: require(false, "unsupported precision");
        }
        grad_output_dims_                   = {a, 1, b, c};
        const std::size_t grad_output_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                              static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t grad_output_bytes = grad_output_count * elementBytes(types[0]);
        host_.resize((grad_output_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_output_count; ++j) { setHostValue(j, 0.125, types[0]); }
        checkHip(hipMalloc(&dgrad_output_, grad_output_bytes), "malloc grad_output");
        checkHip(hipMemcpy(dgrad_output_, host_.data(), grad_output_bytes, hipMemcpyHostToDevice), "fill grad_output");
        view_grad_output_                = dense(dgrad_output_, grad_output_dims_, types[0]);
        residual_dims_                   = {a, 1, b, c};
        const std::size_t residual_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                           static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t residual_bytes = residual_count * elementBytes(types[1]);
        host_.resize((residual_bytes + 7) / 8);
        for (std::size_t j = 0; j < residual_count; ++j) { setHostValue(j, 0.5, types[1]); }
        checkHip(hipMalloc(&dresidual_, residual_bytes), "malloc residual");
        checkHip(hipMemcpy(dresidual_, host_.data(), residual_bytes, hipMemcpyHostToDevice), "fill residual");
        view_residual_ = dense(dresidual_, residual_dims_, types[1]);
        x_dims_        = {a, 1, c};
        const std::size_t x_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(c);
        const std::size_t x_bytes = x_count * elementBytes(types[2]);
        host_.resize((x_bytes + 7) / 8);
        for (std::size_t j = 0; j < x_count; ++j) { setHostValue(j, 0.5, types[2]); }
        checkHip(hipMalloc(&dx_, x_bytes), "malloc x");
        checkHip(hipMemcpy(dx_, host_.data(), x_bytes, hipMemcpyHostToDevice), "fill x");
        view_x_                      = dense(dx_, x_dims_, types[2]);
        bias_dims_                   = {c};
        const std::size_t bias_count = static_cast<std::size_t>(c);
        const std::size_t bias_bytes = bias_count * elementBytes(types[3]);
        host_.resize((bias_bytes + 7) / 8);
        for (std::size_t j = 0; j < bias_count; ++j) { setHostValue(j, 0.25, types[3]); }
        checkHip(hipMalloc(&dbias_, bias_bytes), "malloc bias");
        checkHip(hipMemcpy(dbias_, host_.data(), bias_bytes, hipMemcpyHostToDevice), "fill bias");
        view_bias_                    = dense(dbias_, bias_dims_, types[3]);
        h_res_dims_                   = {a, 1, b, b};
        const std::size_t h_res_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                        static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t h_res_bytes = h_res_count * elementBytes(types[4]);
        host_.resize((h_res_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_res_count; ++j) { setHostValue(j, 0.25, types[4]); }
        checkHip(hipMalloc(&dh_res_, h_res_bytes), "malloc h_res");
        checkHip(hipMemcpy(dh_res_, host_.data(), h_res_bytes, hipMemcpyHostToDevice), "fill h_res");
        view_h_res_  = dense(dh_res_, h_res_dims_, types[4]);
        h_post_dims_ = {a, 1, b};
        const std::size_t h_post_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(b);
        const std::size_t h_post_bytes = h_post_count * elementBytes(types[5]);
        host_.resize((h_post_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_post_count; ++j) { setHostValue(j, 0.5, types[5]); }
        checkHip(hipMalloc(&dh_post_, h_post_bytes), "malloc h_post");
        checkHip(hipMemcpy(dh_post_, host_.data(), h_post_bytes, hipMemcpyHostToDevice), "fill h_post");
        view_h_post_                          = dense(dh_post_, h_post_dims_, types[5]);
        grad_residual_dims_                   = {a, 1, b, c};
        const std::size_t grad_residual_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                                static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t grad_residual_bytes = grad_residual_count * elementBytes(types[6]);
        host_.resize((grad_residual_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_residual_count; ++j) { setHostValue(j, 0.125, types[6]); }
        checkHip(hipMalloc(&dgrad_residual_, grad_residual_bytes), "malloc grad_residual");
        checkHip(hipMemcpy(dgrad_residual_, host_.data(), grad_residual_bytes, hipMemcpyHostToDevice),
                 "fill grad_residual");
        view_grad_residual_ = dense(dgrad_residual_, grad_residual_dims_, types[6]);
        grad_x_dims_        = {a, 1, c};
        const std::size_t grad_x_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(c);
        const std::size_t grad_x_bytes = grad_x_count * elementBytes(types[7]);
        host_.resize((grad_x_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_x_count; ++j) { setHostValue(j, 0.125, types[7]); }
        checkHip(hipMalloc(&dgrad_x_, grad_x_bytes), "malloc grad_x");
        checkHip(hipMemcpy(dgrad_x_, host_.data(), grad_x_bytes, hipMemcpyHostToDevice), "fill grad_x");
        view_grad_x_                       = dense(dgrad_x_, grad_x_dims_, types[7]);
        grad_h_res_dims_                   = {a, 1, b, b};
        const std::size_t grad_h_res_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                             static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t grad_h_res_bytes = grad_h_res_count * elementBytes(types[8]);
        host_.resize((grad_h_res_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_h_res_count; ++j) { setHostValue(j, 0.125, types[8]); }
        checkHip(hipMalloc(&dgrad_h_res_, grad_h_res_bytes), "malloc grad_h_res");
        checkHip(hipMemcpy(dgrad_h_res_, host_.data(), grad_h_res_bytes, hipMemcpyHostToDevice), "fill grad_h_res");
        view_grad_h_res_  = dense(dgrad_h_res_, grad_h_res_dims_, types[8]);
        grad_h_post_dims_ = {a, 1, b};
        const std::size_t grad_h_post_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(b);
        const std::size_t grad_h_post_bytes = grad_h_post_count * elementBytes(types[9]);
        host_.resize((grad_h_post_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_h_post_count; ++j) { setHostValue(j, 0.125, types[9]); }
        checkHip(hipMalloc(&dgrad_h_post_, grad_h_post_bytes), "malloc grad_h_post");
        checkHip(hipMemcpy(dgrad_h_post_, host_.data(), grad_h_post_bytes, hipMemcpyHostToDevice), "fill grad_h_post");
        view_grad_h_post_                 = dense(dgrad_h_post_, grad_h_post_dims_, types[9]);
        grad_bias_dims_                   = {c};
        const std::size_t grad_bias_count = static_cast<std::size_t>(c);
        const std::size_t grad_bias_bytes = grad_bias_count * elementBytes(types[10]);
        host_.resize((grad_bias_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_bias_count; ++j) { setHostValue(j, 0.125, types[10]); }
        checkHip(hipMalloc(&dgrad_bias_, grad_bias_bytes), "malloc grad_bias");
        checkHip(hipMemcpy(dgrad_bias_, host_.data(), grad_bias_bytes, hipMemcpyHostToDevice), "fill grad_bias");
        view_grad_bias_ = dense(dgrad_bias_, grad_bias_dims_, types[10]);
    }
    ~HPostBdaBwdSession() override {
        hipFree(dgrad_output_);
        hipFree(dresidual_);
        hipFree(dx_);
        hipFree(dbias_);
        hipFree(dh_res_);
        hipFree(dh_post_);
        hipFree(dgrad_residual_);
        hipFree(dgrad_x_);
        hipFree(dgrad_h_res_);
        hipFree(dgrad_h_post_);
        hipFree(dgrad_bias_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainHPostBdaBwd(view_grad_output_, view_residual_, view_x_, view_bias_, view_h_res_, view_h_post_,
                                      view_grad_residual_, view_grad_x_, view_grad_h_res_, view_grad_h_post_,
                                      view_grad_bias_, compute_, nullptr, 0, stream),
                    "ftrainHPostBdaBwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_output_), "add grad_output");
        checkFtrain(ftrainPatternAddTensor(pattern, &tresidual_), "add residual");
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tbias_), "add bias");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_res_), "add h_res");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_post_), "add h_post");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_residual_), "add grad_residual");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_x_), "add grad_x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_h_res_), "add grad_h_res");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_h_post_), "add grad_h_post");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_bias_), "add grad_bias");
        checkFtrain(
            ftrainPatternAddHPostBdaBwd(pattern, &operation_, tgrad_output_, tresidual_, tx_, tbias_, th_res_, th_post_,
                                        tgrad_residual_, tgrad_x_, tgrad_h_res_, tgrad_h_post_, tgrad_bias_),
            "ftrainPatternAddHPostBdaBwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tgrad_output_, view_grad_output_), "set grad_output");
        checkFtrain(ftrainArgsSetTensor(args, tresidual_, view_residual_), "set residual");
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, tbias_, view_bias_), "set bias");
        checkFtrain(ftrainArgsSetTensor(args, th_res_, view_h_res_), "set h_res");
        checkFtrain(ftrainArgsSetTensor(args, th_post_, view_h_post_), "set h_post");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_residual_, view_grad_residual_), "set grad_residual");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_x_, view_grad_x_), "set grad_x");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_h_res_, view_grad_h_res_), "set grad_h_res");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_h_post_, view_grad_h_post_), "set grad_h_post");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_bias_, view_grad_bias_), "set grad_bias");
        checkFtrain(ftrainArgsSetHPostBdaBwd(args, operation_, compute_), "ftrainArgsSetHPostBdaBwd");
    }

  private:
    static FTrainStorageView dense(void* memory, const std::vector<std::int64_t>& dims, FTrainNumericType type) {
        return {
            memory, dims.data(), nullptr, static_cast<std::uint8_t>(dims.size()), type, FTRAIN_INDEX_TYPE_CONTINUOUS,
            false};
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
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> grad_output_dims_;
    void* dgrad_output_ = nullptr;
    std::vector<std::int64_t> residual_dims_;
    void* dresidual_ = nullptr;
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> bias_dims_;
    void* dbias_ = nullptr;
    std::vector<std::int64_t> h_res_dims_;
    void* dh_res_ = nullptr;
    std::vector<std::int64_t> h_post_dims_;
    void* dh_post_ = nullptr;
    std::vector<std::int64_t> grad_residual_dims_;
    void* dgrad_residual_ = nullptr;
    std::vector<std::int64_t> grad_x_dims_;
    void* dgrad_x_ = nullptr;
    std::vector<std::int64_t> grad_h_res_dims_;
    void* dgrad_h_res_ = nullptr;
    std::vector<std::int64_t> grad_h_post_dims_;
    void* dgrad_h_post_ = nullptr;
    std::vector<std::int64_t> grad_bias_dims_;
    void* dgrad_bias_ = nullptr;
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_grad_output_{};
    FTrainTensorId tgrad_output_{};
    FTrainStorageView view_residual_{};
    FTrainTensorId tresidual_{};
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_bias_{};
    FTrainTensorId tbias_{};
    FTrainStorageView view_h_res_{};
    FTrainTensorId th_res_{};
    FTrainStorageView view_h_post_{};
    FTrainTensorId th_post_{};
    FTrainStorageView view_grad_residual_{};
    FTrainTensorId tgrad_residual_{};
    FTrainStorageView view_grad_x_{};
    FTrainTensorId tgrad_x_{};
    FTrainStorageView view_grad_h_res_{};
    FTrainTensorId tgrad_h_res_{};
    FTrainStorageView view_grad_h_post_{};
    FTrainTensorId tgrad_h_post_{};
    FTrainStorageView view_grad_bias_{};
    FTrainTensorId tgrad_bias_{};
    FTrainHPostBdaBwdOpId operation_{};
};
struct HPostBdaBwdCase final : OpCase {
    const char* name() const override { return "h_post_bda_bwd"; }
    const char* shapeGrammar() const override { return "tokens,streams,width"; }
    const char* convenienceImpl() const override { return "ftrainHPostBdaBwd"; }
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
                "tokens,streams,width: three positive values required");
        return std::make_unique<HPostBdaBwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<HPostBdaBwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
