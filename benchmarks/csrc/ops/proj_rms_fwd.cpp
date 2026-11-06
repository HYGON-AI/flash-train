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

struct ProjRmsFwdSession final : OpSession {
    ProjRmsFwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 5> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            default: require(false, "unsupported precision");
        }
        x_dims_                   = {a, c};
        const std::size_t x_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t x_bytes = x_count * elementBytes(types[0]);
        host_.resize((x_bytes + 7) / 8);
        for (std::size_t j = 0; j < x_count; ++j) { setHostValue(j, 0.5, types[0]); }
        checkHip(hipMalloc(&dx_, x_bytes), "malloc x");
        checkHip(hipMemcpy(dx_, host_.data(), x_bytes, hipMemcpyHostToDevice), "fill x");
        view_x_                        = dense(dx_, x_dims_, types[0]);
        weight_dims_                   = {b, c};
        const std::size_t weight_count = static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t weight_bytes = weight_count * elementBytes(types[1]);
        host_.resize((weight_bytes + 7) / 8);
        for (std::size_t j = 0; j < weight_count; ++j) { setHostValue(j, 0.25, types[1]); }
        checkHip(hipMalloc(&dweight_, weight_bytes), "malloc weight");
        checkHip(hipMemcpy(dweight_, host_.data(), weight_bytes, hipMemcpyHostToDevice), "fill weight");
        view_weight_                 = dense(dweight_, weight_dims_, types[1]);
        proj_dims_                   = {a, b};
        const std::size_t proj_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t proj_bytes = proj_count * elementBytes(types[2]);
        host_.resize((proj_bytes + 7) / 8);
        for (std::size_t j = 0; j < proj_count; ++j) { setHostValue(j, 123, types[2]); }
        checkHip(hipMalloc(&dproj_, proj_bytes), "malloc proj");
        checkHip(hipMemcpy(dproj_, host_.data(), proj_bytes, hipMemcpyHostToDevice), "fill proj");
        view_proj_                = dense(dproj_, proj_dims_, types[2]);
        r_dims_                   = {a, 1};
        const std::size_t r_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1);
        const std::size_t r_bytes = r_count * elementBytes(types[3]);
        host_.resize((r_bytes + 7) / 8);
        for (std::size_t j = 0; j < r_count; ++j) { setHostValue(j, 123, types[3]); }
        checkHip(hipMalloc(&dr_, r_bytes), "malloc r");
        checkHip(hipMemcpy(dr_, host_.data(), r_bytes, hipMemcpyHostToDevice), "fill r");
        view_r_               = dense(dr_, r_dims_, types[3]);
        eps_value_            = static_cast<float>(1e-6);
        view_eps_             = scalar(&eps_value_, types[4]);
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
    ~ProjRmsFwdSession() override {
        hipFree(dx_);
        hipFree(dweight_);
        hipFree(dproj_);
        hipFree(dr_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainProjRmsFwd(view_x_, view_weight_, view_proj_, view_r_, view_eps_, compute_, workspace_,
                                     workspace_bytes_, stream),
                    "ftrainProjRmsFwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tweight_), "add weight");
        checkFtrain(ftrainPatternAddTensor(pattern, &tproj_), "add proj");
        checkFtrain(ftrainPatternAddTensor(pattern, &tr_), "add r");
        checkFtrain(ftrainPatternAddTensor(pattern, &teps_), "add eps");
        checkFtrain(ftrainPatternAddProjRmsFwd(pattern, &operation_, tx_, tweight_, tproj_, tr_, teps_),
                    "ftrainPatternAddProjRmsFwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, tweight_, view_weight_), "set weight");
        checkFtrain(ftrainArgsSetTensor(args, tproj_, view_proj_), "set proj");
        checkFtrain(ftrainArgsSetTensor(args, tr_, view_r_), "set r");
        checkFtrain(ftrainArgsSetTensor(args, teps_, view_eps_), "set eps");
        checkFtrain(ftrainArgsSetProjRmsFwd(args, operation_, compute_), "ftrainArgsSetProjRmsFwd");
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
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> weight_dims_;
    void* dweight_ = nullptr;
    std::vector<std::int64_t> proj_dims_;
    void* dproj_ = nullptr;
    std::vector<std::int64_t> r_dims_;
    void* dr_ = nullptr;
    float eps_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_weight_{};
    FTrainTensorId tweight_{};
    FTrainStorageView view_proj_{};
    FTrainTensorId tproj_{};
    FTrainStorageView view_r_{};
    FTrainTensorId tr_{};
    FTrainStorageView view_eps_{};
    FTrainTensorId teps_{};
    FTrainProjRmsFwdOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct ProjRmsFwdCase final : OpCase {
    const char* name() const override { return "proj_rms_fwd"; }
    const char* shapeGrammar() const override { return "rows,outputs,reduction"; }
    const char* convenienceImpl() const override { return "ftrainProjRmsFwd"; }
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
        return std::make_unique<ProjRmsFwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<ProjRmsFwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
