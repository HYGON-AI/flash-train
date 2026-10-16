// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct CompactifySession final : OpSession {
    CompactifySession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        require(compute == FTRAIN_NUMERIC_TYPE_FP32, "unsupported precision");
        indices_dims_                   = {a, b};
        const std::size_t indices_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t indices_bytes = indices_count * sizeof(std::int32_t);
        host_.resize(indices_count);
        for (std::size_t j = 0; j < indices_count; ++j) {
            host_[j] = static_cast<std::int32_t>(j % b % c == 0 ? -1 : j % b);
        }
        checkHip(hipMalloc(&dindices_, indices_bytes), "malloc indices");
        checkHip(hipMemcpy(dindices_, host_.data(), indices_bytes, hipMemcpyHostToDevice), "fill indices");
        view_indices_                  = dense(dindices_, indices_dims_);
        output_dims_                   = {a, b};
        const std::size_t output_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t output_bytes = output_count * sizeof(std::int32_t);
        host_.assign(output_count, static_cast<std::int32_t>(0));
        checkHip(hipMalloc(&doutput_, output_bytes), "malloc output");
        checkHip(hipMemcpy(doutput_, host_.data(), output_bytes, hipMemcpyHostToDevice), "fill output");
        view_output_                        = dense(doutput_, output_dims_);
        topk_length_dims_                   = {a};
        const std::size_t topk_length_count = static_cast<std::size_t>(a);
        const std::size_t topk_length_bytes = topk_length_count * sizeof(std::int32_t);
        host_.assign(topk_length_count, static_cast<std::int32_t>(123));
        checkHip(hipMalloc(&dtopk_length_, topk_length_bytes), "malloc topk_length");
        checkHip(hipMemcpy(dtopk_length_, host_.data(), topk_length_bytes, hipMemcpyHostToDevice), "fill topk_length");
        view_topk_length_ = dense(dtopk_length_, topk_length_dims_);
    }
    ~CompactifySession() override {
        hipFree(dindices_);
        hipFree(doutput_);
        hipFree(dtopk_length_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainCompactify(view_indices_, view_output_, view_topk_length_, nullptr, 0, stream),
                    "ftrainCompactify");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tindices_), "add indices");
        checkFtrain(ftrainPatternAddTensor(pattern, &toutput_), "add output");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttopk_length_), "add topk_length");
        checkFtrain(ftrainPatternAddCompactify(pattern, &operation_, tindices_, toutput_, ttopk_length_),
                    "ftrainPatternAddCompactify");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tindices_, view_indices_), "set indices");
        checkFtrain(ftrainArgsSetTensor(args, toutput_, view_output_), "set output");
        checkFtrain(ftrainArgsSetTensor(args, ttopk_length_, view_topk_length_), "set topk_length");
        checkFtrain(ftrainArgsSetCompactify(args, operation_), "ftrainArgsSetCompactify");
    }

  private:
    static FTrainStorageView dense(void* memory, const std::vector<std::int64_t>& dims) {
        return {memory,
                dims.data(),
                nullptr,
                static_cast<std::uint8_t>(dims.size()),
                FTRAIN_NUMERIC_TYPE_INT32,
                FTRAIN_INDEX_TYPE_CONTINUOUS,
                false};
    }
    std::vector<std::int64_t> indices_dims_;
    void* dindices_ = nullptr;
    std::vector<std::int64_t> output_dims_;
    void* doutput_ = nullptr;
    std::vector<std::int64_t> topk_length_dims_;
    void* dtopk_length_ = nullptr;
    std::vector<std::int32_t> host_;
    FTrainStorageView view_indices_{};
    FTrainTensorId tindices_{};
    FTrainStorageView view_output_{};
    FTrainTensorId toutput_{};
    FTrainStorageView view_topk_length_{};
    FTrainTensorId ttopk_length_{};
    FTrainCompactifyOpId operation_{};
};
struct CompactifyCase final : OpCase {
    const char* name() const override { return "compactify"; }
    const char* shapeGrammar() const override { return "rows,columns,invalid_stride"; }
    const char* convenienceImpl() const override { return "ftrainCompactify"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"int32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "rows,columns,invalid_stride: three positive values required");
        return std::make_unique<CompactifySession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<CompactifyCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
