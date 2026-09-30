// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// GEMM 基准接入的参考实现。形状语义 (m, k, n)：a 为 m×k，b 为 k×n，d 为 m×n；
// c 置空（beta=0 不读），alpha/beta 为主机标量，布局统一 CONTINUOUS。

#include <algorithm>
#include <cstdint>
#include <vector>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct GemmSession final : OpSession {
    GemmSession(const std::vector<int64_t>& shape, FTrainNumericType compute)
        : compute_(compute), a_dims_{shape[0], shape[1]}, b_dims_{shape[1], shape[2]},
          cd_dims_{shape[0], shape[2]} {
        const std::int64_t a = shape[0] * shape[1];
        const std::int64_t b = shape[1] * shape[2];
        const std::int64_t d = shape[0] * shape[2];
        host_.assign(static_cast<size_t>(std::max({a, b, d})), 0.5f);
        checkHip(hipMalloc(&da_, static_cast<size_t>(a) * sizeof(float)), "malloc a");
        checkHip(hipMalloc(&db_, static_cast<size_t>(b) * sizeof(float)), "malloc b");
        checkHip(hipMalloc(&dd_, static_cast<size_t>(d) * sizeof(float)), "malloc d");
        checkHip(hipMemcpy(da_, host_.data(), static_cast<size_t>(a) * sizeof(float),
                           hipMemcpyHostToDevice), "fill a");
        checkHip(hipMemcpy(db_, host_.data(), static_cast<size_t>(b) * sizeof(float),
                           hipMemcpyHostToDevice), "fill b");
        checkHip(hipMemcpy(dd_, host_.data(), static_cast<size_t>(d) * sizeof(float),
                           hipMemcpyHostToDevice), "fill d");
        view_a_ = dense2d(da_, a_dims_.data());
        view_b_ = dense2d(db_, b_dims_.data());
        view_c_ = dense2d(nullptr, cd_dims_.data());
        view_d_ = dense2d(dd_, cd_dims_.data());
        view_alpha_ = scalar(&alpha_value_);
        view_beta_ = scalar(&beta_value_);
    }

    ~GemmSession() override {
        hipFree(da_);
        hipFree(db_);
        hipFree(dd_);
    }

    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainGemm(view_a_, view_b_, view_c_, view_d_, view_alpha_, view_beta_,
                               compute_, nullptr, 0, stream),
                    "ftrainGemm");
    }

    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &ta_), "add a");
        checkFtrain(ftrainPatternAddTensor(pattern, &tb_), "add b");
        checkFtrain(ftrainPatternAddTensor(pattern, &tc_), "add c");
        checkFtrain(ftrainPatternAddTensor(pattern, &td_), "add d");
        checkFtrain(ftrainPatternAddTensor(pattern, &talpha_), "add alpha");
        checkFtrain(ftrainPatternAddTensor(pattern, &tbeta_), "add beta");
        checkFtrain(ftrainPatternAddGemm(pattern, &gemm_, ta_, tb_, tc_, td_, talpha_, tbeta_),
                    "add gemm");
    }

    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, ta_, view_a_), "set a");
        checkFtrain(ftrainArgsSetTensor(args, tb_, view_b_), "set b");
        checkFtrain(ftrainArgsSetTensor(args, tc_, view_c_), "set c");
        checkFtrain(ftrainArgsSetTensor(args, td_, view_d_), "set d");
        checkFtrain(ftrainArgsSetTensor(args, talpha_, view_alpha_), "set alpha");
        checkFtrain(ftrainArgsSetTensor(args, tbeta_, view_beta_), "set beta");
        checkFtrain(ftrainArgsSetGemm(args, gemm_, compute_), "set gemm");
    }

private:
    static FTrainStorageView dense2d(void* memory, const int64_t* dims) {
        return FTrainStorageView{memory, dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    }

    static FTrainStorageView scalar(float* value) {
        return FTrainStorageView{value, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32,
                                 FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    }

    FTrainNumericType compute_;
    std::vector<int64_t> a_dims_, b_dims_, cd_dims_;
    std::vector<float> host_;
    void* da_ = nullptr;
    void* db_ = nullptr;
    void* dd_ = nullptr;
    float alpha_value_ = 1.0f;
    float beta_value_ = 0.0f;
    FTrainStorageView view_a_{}, view_b_{}, view_c_{}, view_d_{}, view_alpha_{}, view_beta_{};
    FTrainTensorId ta_{}, tb_{}, tc_{}, td_{}, talpha_{}, tbeta_{};
    FTrainGemmOpId gemm_{};
};

struct GemmCase final : OpCase {
    const char* name() const override { return "gemm"; }

    const char* shapeGrammar() const override { return "m,k,n"; }

    const char* convenienceImpl() const override { return "ftrainGemm"; }

    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {{"fp32", FTRAIN_NUMERIC_TYPE_FP32}};
    }

    std::unique_ptr<OpSession> create(const std::vector<int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "gemm shape expects three positive values m,k,n");
        return std::make_unique<GemmSession>(shape, compute);
    }
};

const bool kRegistered = [] {
    registerOp(std::make_unique<GemmCase>());
    return true;
}();

}  // namespace
}  // namespace ftrain_bench
