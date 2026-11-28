// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct IndexerTopKThdSession final : OpSession {
    IndexerTopKThdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t batch = 1, kv_tokens = a, top_k = std::min<std::int64_t>(a, 32), ratio = 1;
        std::vector<std::int64_t> segments{a};
        if (shape[0] == 8192 && shape[1] == 1024 && shape[2] == 512) {
            segments = {4096, 4096};
            ratio    = 4;
        }
        if (shape[0] == 8192 && shape[1] == 1024 && shape[2] == 1024) {
            segments = {4096, 4096};
            ratio    = 4;
        }
        if (shape[0] == 12288 && shape[1] == 2048 && shape[2] == 1024) {
            segments = {4096, 8192};
            ratio    = 4;
        }
        if (shape[0] == 32768 && shape[1] == 128 && shape[2] == 128) {
            segments = {16384, 16384};
            ratio    = 128;
        }
        if (shape[0] == 8192 && shape[1] == 4096 && shape[2] == 2048) {
            segments = {4096, 4096};
            ratio    = 1;
        }
        if (shape[0] == 12288 && shape[1] == 8192 && shape[2] == 2048) {
            segments = {4096, 8192};
            ratio    = 1;
        }
        std::vector<std::int64_t> cu_q{0}, cu_k{0};
        std::int64_t max_k = 0;
        for (auto length : segments) {
            cu_q.push_back(cu_q.back() + length);
            cu_k.push_back(cu_k.back() + length / ratio);
            max_k = std::max(max_k, length / ratio);
        }
        kv_tokens = cu_k.back();
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
        view_input_values_                   = dense(dinput_values_, input_values_dims_, FTRAIN_NUMERIC_TYPE_FP32);
        cu_seqlens_q_dims_                   = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_q_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_q_bytes = cu_seqlens_q_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((cu_seqlens_q_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_q_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_INT32); }
        for (std::size_t j = 0; j < cu_q.size(); ++j) { setHostValue(j, cu_q[j], FTRAIN_NUMERIC_TYPE_INT32); }
        checkHip(hipMalloc(&hcu_seqlens_q_, cu_seqlens_q_bytes), "malloc cu_seqlens_q");
        checkHip(hipMemcpy(hcu_seqlens_q_, host_.data(), cu_seqlens_q_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_q");
        view_cu_seqlens_q_  = dense(hcu_seqlens_q_, cu_seqlens_q_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        cu_seqlens_kv_dims_ = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_kv_count =
            static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_kv_bytes = cu_seqlens_kv_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((cu_seqlens_kv_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_kv_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_INT32); }
        for (std::size_t j = 0; j < cu_k.size(); ++j) { setHostValue(j, cu_k[j], FTRAIN_NUMERIC_TYPE_INT32); }
        checkHip(hipMalloc(&hcu_seqlens_kv_, cu_seqlens_kv_bytes), "malloc cu_seqlens_kv");
        checkHip(hipMemcpy(hcu_seqlens_kv_, host_.data(), cu_seqlens_kv_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_kv");
        view_cu_seqlens_kv_    = dense(hcu_seqlens_kv_, cu_seqlens_kv_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        q_causal_offsets_dims_ = {static_cast<std::int64_t>(segments.size())};
        const std::size_t q_causal_offsets_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size()));
        const std::size_t q_causal_offsets_bytes = q_causal_offsets_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((q_causal_offsets_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_causal_offsets_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_INT32); }
        checkHip(hipMalloc(&dq_causal_offsets_, q_causal_offsets_bytes), "malloc q_causal_offsets");
        checkHip(hipMemcpy(dq_causal_offsets_, host_.data(), q_causal_offsets_bytes, hipMemcpyHostToDevice),
                 "fill q_causal_offsets");
        view_q_causal_offsets_ = dense(dq_causal_offsets_, q_causal_offsets_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        out_indices_dims_      = {a, c};
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
        ratio_value_        = static_cast<std::int64_t>(ratio);
        view_ratio_         = scalar(&ratio_value_, FTRAIN_NUMERIC_TYPE_INT64);
        top_k_value_        = static_cast<std::int64_t>(c);
        view_top_k_         = scalar(&top_k_value_, FTRAIN_NUMERIC_TYPE_INT64);
        order_values_value_ = static_cast<std::uint8_t>(1);
        view_order_values_  = scalar(&order_values_value_, FTRAIN_NUMERIC_TYPE_UINT8);
    }
    ~IndexerTopKThdSession() override {
        hipFree(dinput_values_);
        hipFree(hcu_seqlens_q_);
        hipFree(hcu_seqlens_kv_);
        hipFree(dq_causal_offsets_);
        hipFree(dout_indices_);
        hipFree(dout_values_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainIndexerTopKThd(view_input_values_, view_cu_seqlens_q_, view_cu_seqlens_kv_,
                                         view_q_causal_offsets_, view_out_indices_, view_out_values_, view_ratio_,
                                         view_top_k_, view_order_values_, nullptr, 0, stream),
                    "ftrainIndexerTopKThd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tinput_values_), "add input_values");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_q_), "add cu_seqlens_q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_kv_), "add cu_seqlens_kv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_causal_offsets_), "add q_causal_offsets");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_indices_), "add out_indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &tout_values_), "add out_values");
        checkFtrain(ftrainPatternAddTensor(pattern, &tratio_), "add ratio");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttop_k_), "add top_k");
        checkFtrain(ftrainPatternAddTensor(pattern, &torder_values_), "add order_values");
        checkFtrain(ftrainPatternAddIndexerTopKThd(pattern, &operation_, tinput_values_, tcu_seqlens_q_,
                                                   tcu_seqlens_kv_, tq_causal_offsets_, tout_indices_, tout_values_,
                                                   tratio_, ttop_k_, torder_values_),
                    "ftrainPatternAddIndexerTopKThd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tinput_values_, view_input_values_), "set input_values");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_q_, view_cu_seqlens_q_), "set cu_seqlens_q");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_kv_, view_cu_seqlens_kv_), "set cu_seqlens_kv");
        checkFtrain(ftrainArgsSetTensor(args, tq_causal_offsets_, view_q_causal_offsets_), "set q_causal_offsets");
        checkFtrain(ftrainArgsSetTensor(args, tout_indices_, view_out_indices_), "set out_indices");
        checkFtrain(ftrainArgsSetTensor(args, tout_values_, view_out_values_), "set out_values");
        checkFtrain(ftrainArgsSetTensor(args, tratio_, view_ratio_), "set ratio");
        checkFtrain(ftrainArgsSetTensor(args, ttop_k_, view_top_k_), "set top_k");
        checkFtrain(ftrainArgsSetTensor(args, torder_values_, view_order_values_), "set order_values");
        checkFtrain(ftrainArgsSetIndexerTopKThd(args, operation_), "ftrainArgsSetIndexerTopKThd");
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
    std::vector<std::int64_t> cu_seqlens_q_dims_;
    void* hcu_seqlens_q_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_kv_dims_;
    void* hcu_seqlens_kv_ = nullptr;
    std::vector<std::int64_t> q_causal_offsets_dims_;
    void* dq_causal_offsets_ = nullptr;
    std::vector<std::int64_t> out_indices_dims_;
    void* dout_indices_ = nullptr;
    std::vector<std::int64_t> out_values_dims_;
    void* dout_values_ = nullptr;
    std::int64_t ratio_value_{};
    std::int64_t top_k_value_{};
    std::uint8_t order_values_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_input_values_{};
    FTrainTensorId tinput_values_{};
    FTrainStorageView view_cu_seqlens_q_{};
    FTrainTensorId tcu_seqlens_q_{};
    FTrainStorageView view_cu_seqlens_kv_{};
    FTrainTensorId tcu_seqlens_kv_{};
    FTrainStorageView view_q_causal_offsets_{};
    FTrainTensorId tq_causal_offsets_{};
    FTrainStorageView view_out_indices_{};
    FTrainTensorId tout_indices_{};
    FTrainStorageView view_out_values_{};
    FTrainTensorId tout_values_{};
    FTrainStorageView view_ratio_{};
    FTrainTensorId tratio_{};
    FTrainStorageView view_top_k_{};
    FTrainTensorId ttop_k_{};
    FTrainStorageView view_order_values_{};
    FTrainTensorId torder_values_{};
    FTrainIndexerTopKThdOpId operation_{};
};
struct IndexerTopKThdCase final : OpCase {
    const char* name() const override { return "indexer_topk_thd"; }
    const char* shapeGrammar() const override { return "total_queries,max_kv_length,top_k"; }
    const char* convenienceImpl() const override { return "ftrainIndexerTopKThd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "total_queries,max_kv_length,top_k: three positive values required");
        return std::make_unique<IndexerTopKThdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<IndexerTopKThdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
