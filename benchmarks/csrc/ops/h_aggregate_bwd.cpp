// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct HAggregateBwdSession final : OpSession {
    HAggregateBwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        require(compute == FTRAIN_NUMERIC_TYPE_FP32, "unsupported precision");
        grad_aggregated_dims_ = {a, 1, c};
        const std::size_t grad_aggregated_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(c);
        const std::size_t grad_aggregated_bytes = grad_aggregated_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((grad_aggregated_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_aggregated_count; ++j) { setHostValue(j, 0.125, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dgrad_aggregated_, grad_aggregated_bytes), "malloc grad_aggregated");
        checkHip(hipMemcpy(dgrad_aggregated_, host_.data(), grad_aggregated_bytes, hipMemcpyHostToDevice),
                 "fill grad_aggregated");
        view_grad_aggregated_     = dense(dgrad_aggregated_, grad_aggregated_dims_);
        x_dims_                   = {a, 1, b, c};
        const std::size_t x_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                    static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t x_bytes = x_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((x_bytes + 7) / 8);
        for (std::size_t j = 0; j < x_count; ++j) { setHostValue(j, 0.5, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dx_, x_bytes), "malloc x");
        checkHip(hipMemcpy(dx_, host_.data(), x_bytes, hipMemcpyHostToDevice), "fill x");
        view_x_     = dense(dx_, x_dims_);
        h_pre_dims_ = {a, 1, b};
        const std::size_t h_pre_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(b);
        const std::size_t h_pre_bytes = h_pre_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((h_pre_bytes + 7) / 8);
        for (std::size_t j = 0; j < h_pre_count; ++j) { setHostValue(j, 0.25, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dh_pre_, h_pre_bytes), "malloc h_pre");
        checkHip(hipMemcpy(dh_pre_, host_.data(), h_pre_bytes, hipMemcpyHostToDevice), "fill h_pre");
        view_h_pre_                    = dense(dh_pre_, h_pre_dims_);
        grad_x_dims_                   = {a, 1, b, c};
        const std::size_t grad_x_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(1) *
                                         static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t grad_x_bytes = grad_x_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((grad_x_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_x_count; ++j) { setHostValue(j, 0.125, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dgrad_x_, grad_x_bytes), "malloc grad_x");
        checkHip(hipMemcpy(dgrad_x_, host_.data(), grad_x_bytes, hipMemcpyHostToDevice), "fill grad_x");
        view_grad_x_     = dense(dgrad_x_, grad_x_dims_);
        grad_h_pre_dims_ = {a, 1, b};
        const std::size_t grad_h_pre_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(b);
        const std::size_t grad_h_pre_bytes = grad_h_pre_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((grad_h_pre_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_h_pre_count; ++j) { setHostValue(j, 0.125, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dgrad_h_pre_, grad_h_pre_bytes), "malloc grad_h_pre");
        checkHip(hipMemcpy(dgrad_h_pre_, host_.data(), grad_h_pre_bytes, hipMemcpyHostToDevice), "fill grad_h_pre");
        view_grad_h_pre_ = dense(dgrad_h_pre_, grad_h_pre_dims_);
    }
    ~HAggregateBwdSession() override {
        hipFree(dgrad_aggregated_);
        hipFree(dx_);
        hipFree(dh_pre_);
        hipFree(dgrad_x_);
        hipFree(dgrad_h_pre_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainHAggregateBwd(view_grad_aggregated_, view_x_, view_h_pre_, view_grad_x_, view_grad_h_pre_,
                                        compute_, nullptr, 0, stream),
                    "ftrainHAggregateBwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_aggregated_), "add grad_aggregated");
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_pre_), "add h_pre");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_x_), "add grad_x");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_h_pre_), "add grad_h_pre");
        checkFtrain(ftrainPatternAddHAggregateBwd(pattern, &operation_, tgrad_aggregated_, tx_, th_pre_, tgrad_x_,
                                                  tgrad_h_pre_),
                    "ftrainPatternAddHAggregateBwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tgrad_aggregated_, view_grad_aggregated_), "set grad_aggregated");
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, th_pre_, view_h_pre_), "set h_pre");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_x_, view_grad_x_), "set grad_x");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_h_pre_, view_grad_h_pre_), "set grad_h_pre");
        checkFtrain(ftrainArgsSetHAggregateBwd(args, operation_, compute_), "ftrainArgsSetHAggregateBwd");
    }

  private:
    static FTrainStorageView dense(void* memory, const std::vector<std::int64_t>& dims) {
        return {memory,
                dims.data(),
                nullptr,
                static_cast<std::uint8_t>(dims.size()),
                FTRAIN_NUMERIC_TYPE_FP32,
                FTRAIN_INDEX_TYPE_CONTINUOUS,
                false};
    }
    static std::size_t elementBytes(FTrainNumericType type) { return 4; }
    void setHostValue(std::size_t index, double value, FTrainNumericType type) {
        auto* dst     = reinterpret_cast<unsigned char*>(host_.data()) + index * elementBytes(type);
        const float v = static_cast<float>(value);
        std::memcpy(dst, &v, sizeof(v));
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> grad_aggregated_dims_;
    void* dgrad_aggregated_ = nullptr;
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> h_pre_dims_;
    void* dh_pre_ = nullptr;
    std::vector<std::int64_t> grad_x_dims_;
    void* dgrad_x_ = nullptr;
    std::vector<std::int64_t> grad_h_pre_dims_;
    void* dgrad_h_pre_ = nullptr;
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_grad_aggregated_{};
    FTrainTensorId tgrad_aggregated_{};
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_h_pre_{};
    FTrainTensorId th_pre_{};
    FTrainStorageView view_grad_x_{};
    FTrainTensorId tgrad_x_{};
    FTrainStorageView view_grad_h_pre_{};
    FTrainTensorId tgrad_h_pre_{};
    FTrainHAggregateBwdOpId operation_{};
};
struct HAggregateBwdCase final : OpCase {
    const char* name() const override { return "h_aggregate_bwd"; }
    const char* shapeGrammar() const override { return "tokens,streams,width"; }
    const char* convenienceImpl() const override { return "ftrainHAggregateBwd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "tokens,streams,width: three positive values required");
        return std::make_unique<HAggregateBwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<HAggregateBwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
