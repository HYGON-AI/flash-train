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

struct IndexerTopKSession final : OpSession {
    IndexerTopKSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        require(compute == FTRAIN_NUMERIC_TYPE_FP32, "unsupported precision");
        input_values_dims_                   = {a, b};
        const std::size_t input_values_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t input_values_bytes = input_values_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((input_values_bytes + 7) / 8);
        for (std::size_t j = 0; j < input_values_count; ++j) {
            setHostValue(j, (j % b) * 0.01, FTRAIN_NUMERIC_TYPE_FP32);
        }
        checkHip(hipMalloc(&dinput_values_, input_values_bytes), "malloc input_values");
        checkHip(hipMemcpy(dinput_values_, host_.data(), input_values_bytes, hipMemcpyHostToDevice),
                 "fill input_values");
        view_input_values_               = dense(dinput_values_, input_values_dims_, FTRAIN_NUMERIC_TYPE_FP32);
        seq_lens_dims_                   = {a};
        const std::size_t seq_lens_count = static_cast<std::size_t>(a);
        const std::size_t seq_lens_bytes = seq_lens_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((seq_lens_bytes + 7) / 8);
        for (std::size_t j = 0; j < seq_lens_count; ++j) { setHostValue(j, b, FTRAIN_NUMERIC_TYPE_INT32); }
        checkHip(hipMalloc(&dseq_lens_, seq_lens_bytes), "malloc seq_lens");
        checkHip(hipMemcpy(dseq_lens_, host_.data(), seq_lens_bytes, hipMemcpyHostToDevice), "fill seq_lens");
        view_seq_lens_                      = dense(dseq_lens_, seq_lens_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        out_indices_dims_                   = {a, c};
        const std::size_t out_indices_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t out_indices_bytes = out_indices_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((out_indices_bytes + 7) / 8);
        for (std::size_t j = 0; j < out_indices_count; ++j) { setHostValue(j, 123, FTRAIN_NUMERIC_TYPE_INT32); }
        checkHip(hipMalloc(&dout_indices_, out_indices_bytes), "malloc out_indices");
        checkHip(hipMemcpy(dout_indices_, host_.data(), out_indices_bytes, hipMemcpyHostToDevice), "fill out_indices");
        view_out_indices_                  = dense(dout_indices_, out_indices_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        out_values_dims_                   = {a, c};
        const std::size_t out_values_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t out_values_bytes = out_values_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((out_values_bytes + 7) / 8);
        for (std::size_t j = 0; j < out_values_count; ++j) { setHostValue(j, 123, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dout_values_, out_values_bytes), "malloc out_values");
        checkHip(hipMemcpy(dout_values_, host_.data(), out_values_bytes, hipMemcpyHostToDevice), "fill out_values");
        view_out_values_    = dense(dout_values_, out_values_dims_, FTRAIN_NUMERIC_TYPE_FP32);
        top_k_value_        = static_cast<std::int64_t>(c);
        view_top_k_         = scalar(&top_k_value_, FTRAIN_NUMERIC_TYPE_INT64);
        next_n_value_       = static_cast<std::int64_t>(1);
        view_next_n_        = scalar(&next_n_value_, FTRAIN_NUMERIC_TYPE_INT64);
        order_values_value_ = static_cast<std::uint8_t>(1);
        view_order_values_  = scalar(&order_values_value_, FTRAIN_NUMERIC_TYPE_UINT8);
    }
    ~IndexerTopKSession() override {
        hipFree(dinput_values_);
        hipFree(dseq_lens_);
        hipFree(dout_indices_);
        hipFree(dout_values_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainIndexerTopK(view_input_values_, view_seq_lens_, view_out_indices_, view_out_values_,
                                      view_top_k_, view_next_n_, view_order_values_, nullptr, 0, stream),
                    "ftrainIndexerTopK");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tinput_values_), "add input_values");
        checkFtrain(ftrainPatternAddTensor(pattern, &tseq_lens_), "add seq_lens");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_indices_), "add out_indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_values_), "add out_values");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttop_k_), "add top_k");
        checkFtrain(ftrainPatternAddTensor(pattern, &tnext_n_), "add next_n");
        checkFtrain(ftrainPatternAddTensor(pattern, &torder_values_), "add order_values");
        checkFtrain(ftrainPatternAddIndexerTopK(pattern, &operation_, tinput_values_, tseq_lens_, tout_indices_,
                                                tout_values_, ttop_k_, tnext_n_, torder_values_),
                    "ftrainPatternAddIndexerTopK");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tinput_values_, view_input_values_), "set input_values");
        checkFtrain(ftrainArgsSetTensor(args, tseq_lens_, view_seq_lens_), "set seq_lens");
        checkFtrain(ftrainArgsSetTensor(args, tout_indices_, view_out_indices_), "set out_indices");
        checkFtrain(ftrainArgsSetTensor(args, tout_values_, view_out_values_), "set out_values");
        checkFtrain(ftrainArgsSetTensor(args, ttop_k_, view_top_k_), "set top_k");
        checkFtrain(ftrainArgsSetTensor(args, tnext_n_, view_next_n_), "set next_n");
        checkFtrain(ftrainArgsSetTensor(args, torder_values_, view_order_values_), "set order_values");
        checkFtrain(ftrainArgsSetIndexerTopK(args, operation_), "ftrainArgsSetIndexerTopK");
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
        return (type == FTRAIN_NUMERIC_TYPE_INT64) ? 8 : ((type == FTRAIN_NUMERIC_TYPE_UINT8) ? 1 : (4));
    }
    void setHostValue(std::size_t index, double value, FTrainNumericType type) {
        auto* dst = reinterpret_cast<unsigned char*>(host_.data()) + index * elementBytes(type);
        if (type == FTRAIN_NUMERIC_TYPE_FP32) {
            const float v = static_cast<float>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT64) {
            const std::int64_t v = static_cast<std::int64_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT32) {
            const std::int32_t v = static_cast<std::int32_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_UINT8) {
            *dst = static_cast<std::uint8_t>(value);
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    std::vector<std::int64_t> input_values_dims_;
    void* dinput_values_ = nullptr;
    std::vector<std::int64_t> seq_lens_dims_;
    void* dseq_lens_ = nullptr;
    std::vector<std::int64_t> out_indices_dims_;
    void* dout_indices_ = nullptr;
    std::vector<std::int64_t> out_values_dims_;
    void* dout_values_ = nullptr;
    std::int64_t top_k_value_{};
    std::int64_t next_n_value_{};
    std::uint8_t order_values_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_input_values_{};
    FTrainTensorId tinput_values_{};
    FTrainStorageView view_seq_lens_{};
    FTrainTensorId tseq_lens_{};
    FTrainStorageView view_out_indices_{};
    FTrainTensorId tout_indices_{};
    FTrainStorageView view_out_values_{};
    FTrainTensorId tout_values_{};
    FTrainStorageView view_top_k_{};
    FTrainTensorId ttop_k_{};
    FTrainStorageView view_next_n_{};
    FTrainTensorId tnext_n_{};
    FTrainStorageView view_order_values_{};
    FTrainTensorId torder_values_{};
    FTrainIndexerTopKOpId operation_{};
};
struct IndexerTopKCase final : OpCase {
    const char* name() const override { return "indexer_topk"; }
    const char* shapeGrammar() const override { return "rows,columns,top_k"; }
    const char* convenienceImpl() const override { return "ftrainIndexerTopK"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "rows,columns,top_k: three positive values required");
        return std::make_unique<IndexerTopKSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<IndexerTopKCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
