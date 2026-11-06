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

struct ProjRmsComputeHFwdSession final : OpSession {
    ProjRmsComputeHFwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 14> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_INT64,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_INT64,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_INT64,
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
        weight_dims_                   = {b * b + 2 * b, c};
        const std::size_t weight_count = static_cast<std::size_t>(b * b + 2 * b) * static_cast<std::size_t>(c);
        const std::size_t weight_bytes = weight_count * elementBytes(types[1]);
        host_.resize((weight_bytes + 7) / 8);
        for (std::size_t j = 0; j < weight_count; ++j) { setHostValue(j, 0.25, types[1]); }
        checkHip(hipMalloc(&dweight_, weight_bytes), "malloc weight");
        checkHip(hipMemcpy(dweight_, host_.data(), weight_bytes, hipMemcpyHostToDevice), "fill weight");
        view_weight_                      = dense(dweight_, weight_dims_, types[1]);
        alpha_pre_dims_                   = {1};
        const std::size_t alpha_pre_count = static_cast<std::size_t>(1);
        const std::size_t alpha_pre_bytes = alpha_pre_count * elementBytes(types[2]);
        host_.resize((alpha_pre_bytes + 7) / 8);
        for (std::size_t j = 0; j < alpha_pre_count; ++j) { setHostValue(j, 1, types[2]); }
        checkHip(hipMalloc(&dalpha_pre_, alpha_pre_bytes), "malloc alpha_pre");
        checkHip(hipMemcpy(dalpha_pre_, host_.data(), alpha_pre_bytes, hipMemcpyHostToDevice), "fill alpha_pre");
        view_alpha_pre_                    = dense(dalpha_pre_, alpha_pre_dims_, types[2]);
        alpha_post_dims_                   = {1};
        const std::size_t alpha_post_count = static_cast<std::size_t>(1);
        const std::size_t alpha_post_bytes = alpha_post_count * elementBytes(types[3]);
        host_.resize((alpha_post_bytes + 7) / 8);
        for (std::size_t j = 0; j < alpha_post_count; ++j) { setHostValue(j, 1, types[3]); }
        checkHip(hipMalloc(&dalpha_post_, alpha_post_bytes), "malloc alpha_post");
        checkHip(hipMemcpy(dalpha_post_, host_.data(), alpha_post_bytes, hipMemcpyHostToDevice), "fill alpha_post");
        view_alpha_post_                  = dense(dalpha_post_, alpha_post_dims_, types[3]);
        alpha_res_dims_                   = {1};
        const std::size_t alpha_res_count = static_cast<std::size_t>(1);
        const std::size_t alpha_res_bytes = alpha_res_count * elementBytes(types[4]);
        host_.resize((alpha_res_bytes + 7) / 8);
        for (std::size_t j = 0; j < alpha_res_count; ++j) { setHostValue(j, 1, types[4]); }
        checkHip(hipMalloc(&dalpha_res_, alpha_res_bytes), "malloc alpha_res");
        checkHip(hipMemcpy(dalpha_res_, host_.data(), alpha_res_bytes, hipMemcpyHostToDevice), "fill alpha_res");
        view_alpha_res_              = dense(dalpha_res_, alpha_res_dims_, types[4]);
        bias_dims_                   = {b * b + 2 * b};
        const std::size_t bias_count = static_cast<std::size_t>(b * b + 2 * b);
        const std::size_t bias_bytes = bias_count * elementBytes(types[5]);
        host_.resize((bias_bytes + 7) / 8);
        for (std::size_t j = 0; j < bias_count; ++j) { setHostValue(j, 0, types[5]); }
        checkHip(hipMalloc(&dbias_, bias_bytes), "malloc bias");
        checkHip(hipMemcpy(dbias_, host_.data(), bias_bytes, hipMemcpyHostToDevice), "fill bias");
        view_bias_                    = dense(dbias_, bias_dims_, types[5]);
        h_pre_dims_                   = {a, b};
        const std::size_t h_pre_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t h_pre_bytes = h_pre_count * elementBytes(types[6]);
        host_.resize((h_pre_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_pre_count; ++j) {
            setHostValue(j, 1.0 / (1.0 + std::exp(-c * 0.25)) + 1e-6, types[6]);
        }
        checkHip(hipMalloc(&dh_pre_, h_pre_bytes), "malloc h_pre");
        checkHip(hipMemcpy(dh_pre_, host_.data(), h_pre_bytes, hipMemcpyHostToDevice), "fill h_pre");
        view_h_pre_                    = dense(dh_pre_, h_pre_dims_, types[6]);
        h_post_dims_                   = {a, b};
        const std::size_t h_post_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t h_post_bytes = h_post_count * elementBytes(types[7]);
        host_.resize((h_post_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_post_count; ++j) { setHostValue(j, 2.0 / (1.0 + std::exp(-c * 0.25)), types[7]); }
        checkHip(hipMalloc(&dh_post_, h_post_bytes), "malloc h_post");
        checkHip(hipMemcpy(dh_post_, host_.data(), h_post_bytes, hipMemcpyHostToDevice), "fill h_post");
        view_h_post_                  = dense(dh_post_, h_post_dims_, types[7]);
        h_res_dims_                   = {a, b * b};
        const std::size_t h_res_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b * b);
        const std::size_t h_res_bytes = h_res_count * elementBytes(types[8]);
        host_.resize((h_res_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_res_count; ++j) { setHostValue(j, 123, types[8]); }
        checkHip(hipMalloc(&dh_res_, h_res_bytes), "malloc h_res");
        checkHip(hipMemcpy(dh_res_, host_.data(), h_res_bytes, hipMemcpyHostToDevice), "fill h_res");
        view_h_res_               = dense(dh_res_, h_res_dims_, types[8]);
        r_dims_                   = {a, 1};
        const std::size_t r_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1);
        const std::size_t r_bytes = r_count * elementBytes(types[9]);
        host_.resize((r_bytes + 7) / 8);
        for (std::size_t j = 0; j < r_count; ++j) { setHostValue(j, 0.5, types[9]); }
        checkHip(hipMalloc(&dr_, r_bytes), "malloc r");
        checkHip(hipMemcpy(dr_, host_.data(), r_bytes, hipMemcpyHostToDevice), "fill r");
        view_r_                      = dense(dr_, r_dims_, types[9]);
        proj_dims_                   = {a, b * b + 2 * b};
        const std::size_t proj_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b * b + 2 * b);
        const std::size_t proj_bytes = proj_count * elementBytes(types[10]);
        host_.resize((proj_bytes + 7) / 8);
        for (std::size_t j = 0; j < proj_count; ++j) { setHostValue(j, c * 0.125, types[10]); }
        checkHip(hipMalloc(&dproj_, proj_bytes), "malloc proj");
        checkHip(hipMemcpy(dproj_, host_.data(), proj_bytes, hipMemcpyHostToDevice), "fill proj");
        view_proj_            = dense(dproj_, proj_dims_, types[10]);
        n_value_              = static_cast<std::int64_t>(b);
        view_n_               = scalar(&n_value_, types[11]);
        eps_value_            = static_cast<float>(1e-6);
        view_eps_             = scalar(&eps_value_, types[12]);
        compute_h_eps_value_  = static_cast<float>(1e-6);
        view_compute_h_eps_   = scalar(&compute_h_eps_value_, types[13]);
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
    ~ProjRmsComputeHFwdSession() override {
        hipFree(dx_);
        hipFree(dweight_);
        hipFree(dalpha_pre_);
        hipFree(dalpha_post_);
        hipFree(dalpha_res_);
        hipFree(dbias_);
        hipFree(dh_pre_);
        hipFree(dh_post_);
        hipFree(dh_res_);
        hipFree(dr_);
        hipFree(dproj_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(
            ftrainProjRmsComputeHFwd(view_x_, view_weight_, view_alpha_pre_, view_alpha_post_, view_alpha_res_,
                                     view_bias_, view_h_pre_, view_h_post_, view_h_res_, view_r_, view_proj_, view_n_,
                                     view_eps_, view_compute_h_eps_, compute_, workspace_, workspace_bytes_, stream),
            "ftrainProjRmsComputeHFwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tweight_), "add weight");
        checkFtrain(ftrainPatternAddTensor(pattern, &talpha_pre_), "add alpha_pre");
        checkFtrain(ftrainPatternAddTensor(pattern, &talpha_post_), "add alpha_post");
        checkFtrain(ftrainPatternAddTensor(pattern, &talpha_res_), "add alpha_res");
        checkFtrain(ftrainPatternAddTensor(pattern, &tbias_), "add bias");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_pre_), "add h_pre");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_post_), "add h_post");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_res_), "add h_res");
        checkFtrain(ftrainPatternAddTensor(pattern, &tr_), "add r");
        checkFtrain(ftrainPatternAddTensor(pattern, &tproj_), "add proj");
        checkFtrain(ftrainPatternAddTensor(pattern, &tn_), "add n");
        checkFtrain(ftrainPatternAddTensor(pattern, &teps_), "add eps");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcompute_h_eps_), "add compute_h_eps");
        checkFtrain(ftrainPatternAddProjRmsComputeHFwd(pattern, &operation_, tx_, tweight_, talpha_pre_, talpha_post_,
                                                       talpha_res_, tbias_, th_pre_, th_post_, th_res_, tr_, tproj_,
                                                       tn_, teps_, tcompute_h_eps_),
                    "ftrainPatternAddProjRmsComputeHFwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, tweight_, view_weight_), "set weight");
        checkFtrain(ftrainArgsSetTensor(args, talpha_pre_, view_alpha_pre_), "set alpha_pre");
        checkFtrain(ftrainArgsSetTensor(args, talpha_post_, view_alpha_post_), "set alpha_post");
        checkFtrain(ftrainArgsSetTensor(args, talpha_res_, view_alpha_res_), "set alpha_res");
        checkFtrain(ftrainArgsSetTensor(args, tbias_, view_bias_), "set bias");
        checkFtrain(ftrainArgsSetTensor(args, th_pre_, view_h_pre_), "set h_pre");
        checkFtrain(ftrainArgsSetTensor(args, th_post_, view_h_post_), "set h_post");
        checkFtrain(ftrainArgsSetTensor(args, th_res_, view_h_res_), "set h_res");
        checkFtrain(ftrainArgsSetTensor(args, tr_, view_r_), "set r");
        checkFtrain(ftrainArgsSetTensor(args, tproj_, view_proj_), "set proj");
        checkFtrain(ftrainArgsSetTensor(args, tn_, view_n_), "set n");
        checkFtrain(ftrainArgsSetTensor(args, teps_, view_eps_), "set eps");
        checkFtrain(ftrainArgsSetTensor(args, tcompute_h_eps_, view_compute_h_eps_), "set compute_h_eps");
        checkFtrain(ftrainArgsSetProjRmsComputeHFwd(args, operation_, compute_), "ftrainArgsSetProjRmsComputeHFwd");
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
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> weight_dims_;
    void* dweight_ = nullptr;
    std::vector<std::int64_t> alpha_pre_dims_;
    void* dalpha_pre_ = nullptr;
    std::vector<std::int64_t> alpha_post_dims_;
    void* dalpha_post_ = nullptr;
    std::vector<std::int64_t> alpha_res_dims_;
    void* dalpha_res_ = nullptr;
    std::vector<std::int64_t> bias_dims_;
    void* dbias_ = nullptr;
    std::vector<std::int64_t> h_pre_dims_;
    void* dh_pre_ = nullptr;
    std::vector<std::int64_t> h_post_dims_;
    void* dh_post_ = nullptr;
    std::vector<std::int64_t> h_res_dims_;
    void* dh_res_ = nullptr;
    std::vector<std::int64_t> r_dims_;
    void* dr_ = nullptr;
    std::vector<std::int64_t> proj_dims_;
    void* dproj_ = nullptr;
    std::int64_t n_value_{};
    float eps_value_{};
    float compute_h_eps_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_weight_{};
    FTrainTensorId tweight_{};
    FTrainStorageView view_alpha_pre_{};
    FTrainTensorId talpha_pre_{};
    FTrainStorageView view_alpha_post_{};
    FTrainTensorId talpha_post_{};
    FTrainStorageView view_alpha_res_{};
    FTrainTensorId talpha_res_{};
    FTrainStorageView view_bias_{};
    FTrainTensorId tbias_{};
    FTrainStorageView view_h_pre_{};
    FTrainTensorId th_pre_{};
    FTrainStorageView view_h_post_{};
    FTrainTensorId th_post_{};
    FTrainStorageView view_h_res_{};
    FTrainTensorId th_res_{};
    FTrainStorageView view_r_{};
    FTrainTensorId tr_{};
    FTrainStorageView view_proj_{};
    FTrainTensorId tproj_{};
    FTrainStorageView view_n_{};
    FTrainTensorId tn_{};
    FTrainStorageView view_eps_{};
    FTrainTensorId teps_{};
    FTrainStorageView view_compute_h_eps_{};
    FTrainTensorId tcompute_h_eps_{};
    FTrainProjRmsComputeHFwdOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct ProjRmsComputeHFwdCase final : OpCase {
    const char* name() const override { return "proj_rms_compute_h_fwd"; }
    const char* shapeGrammar() const override { return "rows,streams,reduction"; }
    const char* convenienceImpl() const override { return "ftrainProjRmsComputeHFwd"; }
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
                "rows,streams,reduction: three positive values required");
        return std::make_unique<ProjRmsComputeHFwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<ProjRmsComputeHFwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
