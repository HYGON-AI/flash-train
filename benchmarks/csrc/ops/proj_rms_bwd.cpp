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

struct ProjRmsBwdSession final : OpSession {
    ProjRmsBwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 8> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            default: require(false, "unsupported precision");
        }
        grad_proj_dims_                   = {a, b};
        const std::size_t grad_proj_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t grad_proj_bytes = grad_proj_count * elementBytes(types[0]);
        host_.resize((grad_proj_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_proj_count; ++j) { setHostValue(j, 0.125, types[0]); }
        checkHip(hipMalloc(&dgrad_proj_, grad_proj_bytes), "malloc grad_proj");
        checkHip(hipMemcpy(dgrad_proj_, host_.data(), grad_proj_bytes, hipMemcpyHostToDevice), "fill grad_proj");
        view_grad_proj_                = dense(dgrad_proj_, grad_proj_dims_, types[0]);
        grad_r_dims_                   = {a, 1};
        const std::size_t grad_r_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1);
        const std::size_t grad_r_bytes = grad_r_count * elementBytes(types[1]);
        host_.resize((grad_r_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_r_count; ++j) { setHostValue(j, 0.125, types[1]); }
        checkHip(hipMalloc(&dgrad_r_, grad_r_bytes), "malloc grad_r");
        checkHip(hipMemcpy(dgrad_r_, host_.data(), grad_r_bytes, hipMemcpyHostToDevice), "fill grad_r");
        view_grad_r_              = dense(dgrad_r_, grad_r_dims_, types[1]);
        x_dims_                   = {a, c};
        const std::size_t x_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t x_bytes = x_count * elementBytes(types[2]);
        host_.resize((x_bytes + 7) / 8);
        for (std::size_t j = 0; j < x_count; ++j) { setHostValue(j, 0.5, types[2]); }
        checkHip(hipMalloc(&dx_, x_bytes), "malloc x");
        checkHip(hipMemcpy(dx_, host_.data(), x_bytes, hipMemcpyHostToDevice), "fill x");
        view_x_                        = dense(dx_, x_dims_, types[2]);
        weight_dims_                   = {b, c};
        const std::size_t weight_count = static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t weight_bytes = weight_count * elementBytes(types[3]);
        host_.resize((weight_bytes + 7) / 8);
        for (std::size_t j = 0; j < weight_count; ++j) { setHostValue(j, 0.25, types[3]); }
        checkHip(hipMalloc(&dweight_, weight_bytes), "malloc weight");
        checkHip(hipMemcpy(dweight_, host_.data(), weight_bytes, hipMemcpyHostToDevice), "fill weight");
        view_weight_              = dense(dweight_, weight_dims_, types[3]);
        r_dims_                   = {a, 1};
        const std::size_t r_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1);
        const std::size_t r_bytes = r_count * elementBytes(types[4]);
        host_.resize((r_bytes + 7) / 8);
        for (std::size_t j = 0; j < r_count; ++j) { setHostValue(j, 2, types[4]); }
        checkHip(hipMalloc(&dr_, r_bytes), "malloc r");
        checkHip(hipMemcpy(dr_, host_.data(), r_bytes, hipMemcpyHostToDevice), "fill r");
        view_r_                        = dense(dr_, r_dims_, types[4]);
        grad_x_dims_                   = {a, c};
        const std::size_t grad_x_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t grad_x_bytes = grad_x_count * elementBytes(types[5]);
        host_.resize((grad_x_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_x_count; ++j) { setHostValue(j, 0.125, types[5]); }
        checkHip(hipMalloc(&dgrad_x_, grad_x_bytes), "malloc grad_x");
        checkHip(hipMemcpy(dgrad_x_, host_.data(), grad_x_bytes, hipMemcpyHostToDevice), "fill grad_x");
        view_grad_x_                        = dense(dgrad_x_, grad_x_dims_, types[5]);
        grad_weight_dims_                   = {b, c};
        const std::size_t grad_weight_count = static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t grad_weight_bytes = grad_weight_count * elementBytes(types[6]);
        host_.resize((grad_weight_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_weight_count; ++j) { setHostValue(j, 0.125, types[6]); }
        checkHip(hipMalloc(&dgrad_weight_, grad_weight_bytes), "malloc grad_weight");
        checkHip(hipMemcpy(dgrad_weight_, host_.data(), grad_weight_bytes, hipMemcpyHostToDevice), "fill grad_weight");
        view_grad_weight_ = dense(dgrad_weight_, grad_weight_dims_, types[6]);
        eps_value_        = static_cast<float>(1e-6);
        view_eps_         = scalar(&eps_value_, types[7]);
    }
    ~ProjRmsBwdSession() override {
        hipFree(dgrad_proj_);
        hipFree(dgrad_r_);
        hipFree(dx_);
        hipFree(dweight_);
        hipFree(dr_);
        hipFree(dgrad_x_);
        hipFree(dgrad_weight_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainProjRmsBwd(view_grad_proj_, view_grad_r_, view_x_, view_weight_, view_r_, view_grad_x_,
                                     view_grad_weight_, view_eps_, compute_, nullptr, 0, stream),
                    "ftrainProjRmsBwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_proj_), "add grad_proj");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_r_), "add grad_r");
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tweight_), "add weight");
        checkFtrain(ftrainPatternAddTensor(pattern, &tr_), "add r");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_x_), "add grad_x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_weight_), "add grad_weight");
        checkFtrain(ftrainPatternAddTensor(pattern, &teps_), "add eps");
        checkFtrain(ftrainPatternAddProjRmsBwd(pattern, &operation_, tgrad_proj_, tgrad_r_, tx_, tweight_, tr_,
                                               tgrad_x_, tgrad_weight_, teps_),
                    "ftrainPatternAddProjRmsBwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tgrad_proj_, view_grad_proj_), "set grad_proj");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_r_, view_grad_r_), "set grad_r");
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, tweight_, view_weight_), "set weight");
        checkFtrain(ftrainArgsSetTensor(args, tr_, view_r_), "set r");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_x_, view_grad_x_), "set grad_x");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_weight_, view_grad_weight_), "set grad_weight");
        checkFtrain(ftrainArgsSetTensor(args, teps_, view_eps_), "set eps");
        checkFtrain(ftrainArgsSetProjRmsBwd(args, operation_, compute_), "ftrainArgsSetProjRmsBwd");
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
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> grad_proj_dims_;
    void* dgrad_proj_ = nullptr;
    std::vector<std::int64_t> grad_r_dims_;
    void* dgrad_r_ = nullptr;
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> weight_dims_;
    void* dweight_ = nullptr;
    std::vector<std::int64_t> r_dims_;
    void* dr_ = nullptr;
    std::vector<std::int64_t> grad_x_dims_;
    void* dgrad_x_ = nullptr;
    std::vector<std::int64_t> grad_weight_dims_;
    void* dgrad_weight_ = nullptr;
    float eps_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_grad_proj_{};
    FTrainTensorId tgrad_proj_{};
    FTrainStorageView view_grad_r_{};
    FTrainTensorId tgrad_r_{};
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_weight_{};
    FTrainTensorId tweight_{};
    FTrainStorageView view_r_{};
    FTrainTensorId tr_{};
    FTrainStorageView view_grad_x_{};
    FTrainTensorId tgrad_x_{};
    FTrainStorageView view_grad_weight_{};
    FTrainTensorId tgrad_weight_{};
    FTrainStorageView view_eps_{};
    FTrainTensorId teps_{};
    FTrainProjRmsBwdOpId operation_{};
};
struct ProjRmsBwdCase final : OpCase {
    const char* name() const override { return "proj_rms_bwd"; }
    const char* shapeGrammar() const override { return "rows,outputs,reduction"; }
    const char* convenienceImpl() const override { return "ftrainProjRmsBwd"; }
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
                "rows,outputs,reduction: three positive values required");
        return std::make_unique<ProjRmsBwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<ProjRmsBwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
