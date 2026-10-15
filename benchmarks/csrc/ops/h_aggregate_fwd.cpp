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

struct HAggregateFwdSession final : OpSession {
    HAggregateFwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        require(compute == FTRAIN_NUMERIC_TYPE_FP32, "unsupported precision");
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
        view_h_pre_      = dense(dh_pre_, h_pre_dims_);
        aggregated_dims_ = {a, 1, c};
        const std::size_t aggregated_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(1) * static_cast<std::size_t>(c);
        const std::size_t aggregated_bytes = aggregated_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((aggregated_bytes + 7) / 8);
        for (std::size_t j = 0; j < aggregated_count; ++j) { setHostValue(j, 123, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&daggregated_, aggregated_bytes), "malloc aggregated");
        checkHip(hipMemcpy(daggregated_, host_.data(), aggregated_bytes, hipMemcpyHostToDevice), "fill aggregated");
        view_aggregated_ = dense(daggregated_, aggregated_dims_);
    }
    ~HAggregateFwdSession() override {
        hipFree(dx_);
        hipFree(dh_pre_);
        hipFree(daggregated_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainHAggregateFwd(view_x_, view_h_pre_, view_aggregated_, compute_, nullptr, 0, stream),
                    "ftrainHAggregateFwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tx_), "add x");
        checkFtrain(ftrainPatternAddTensor(pattern, &th_pre_), "add h_pre");
        checkFtrain(ftrainPatternAddTensor(pattern, &taggregated_), "add aggregated");
        checkFtrain(ftrainPatternAddHAggregateFwd(pattern, &operation_, tx_, th_pre_, taggregated_),
                    "ftrainPatternAddHAggregateFwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tx_, view_x_), "set x");
        checkFtrain(ftrainArgsSetTensor(args, th_pre_, view_h_pre_), "set h_pre");
        checkFtrain(ftrainArgsSetTensor(args, taggregated_, view_aggregated_), "set aggregated");
        checkFtrain(ftrainArgsSetHAggregateFwd(args, operation_, compute_), "ftrainArgsSetHAggregateFwd");
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
    std::vector<std::int64_t> x_dims_;
    void* dx_ = nullptr;
    std::vector<std::int64_t> h_pre_dims_;
    void* dh_pre_ = nullptr;
    std::vector<std::int64_t> aggregated_dims_;
    void* daggregated_ = nullptr;
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_x_{};
    FTrainTensorId tx_{};
    FTrainStorageView view_h_pre_{};
    FTrainTensorId th_pre_{};
    FTrainStorageView view_aggregated_{};
    FTrainTensorId taggregated_{};
    FTrainHAggregateFwdOpId operation_{};
};
struct HAggregateFwdCase final : OpCase {
    const char* name() const override { return "h_aggregate_fwd"; }
    const char* shapeGrammar() const override { return "tokens,streams,width"; }
    const char* convenienceImpl() const override { return "ftrainHAggregateFwd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "tokens,streams,width: three positive values required");
        return std::make_unique<HAggregateFwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<HAggregateFwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
