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

struct SinkhornBwdSession final : OpSession {
    SinkhornBwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 5> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            default: require(false, "unsupported precision");
        }
        grad_output_dims_ = {a, b, b};
        const std::size_t grad_output_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t grad_output_bytes = grad_output_count * elementBytes(types[0]);
        host_.resize((grad_output_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_output_count; ++j) { setHostValue(j, 0.125, types[0]); }
        checkHip(hipMalloc(&dgrad_output_, grad_output_bytes), "malloc grad_output");
        checkHip(hipMemcpy(dgrad_output_, host_.data(), grad_output_bytes, hipMemcpyHostToDevice), "fill grad_output");
        view_grad_output_ = dense(dgrad_output_, grad_output_dims_, types[0]);
        m_init_dims_      = {a, b, b};
        const std::size_t m_init_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t m_init_bytes = m_init_count * elementBytes(types[1]);
        host_.resize((m_init_bytes + 7) / 8);
        for (std::size_t j = 0; j < m_init_count; ++j) { setHostValue(j, 1.0 / b + 1e-6, types[1]); }
        checkHip(hipMalloc(&dm_init_, m_init_bytes), "malloc m_init");
        checkHip(hipMemcpy(dm_init_, host_.data(), m_init_bytes, hipMemcpyHostToDevice), "fill m_init");
        view_m_init_      = dense(dm_init_, m_init_dims_, types[1]);
        grad_logits_dims_ = {a, b, b};
        const std::size_t grad_logits_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t grad_logits_bytes = grad_logits_count * elementBytes(types[2]);
        host_.resize((grad_logits_bytes + 7) / 8);
        for (std::size_t j = 0; j < grad_logits_count; ++j) { setHostValue(j, 0.125, types[2]); }
        checkHip(hipMalloc(&dgrad_logits_, grad_logits_bytes), "malloc grad_logits");
        checkHip(hipMemcpy(dgrad_logits_, host_.data(), grad_logits_bytes, hipMemcpyHostToDevice), "fill grad_logits");
        view_grad_logits_     = dense(dgrad_logits_, grad_logits_dims_, types[2]);
        num_iterations_value_ = static_cast<std::int64_t>(c);
        view_num_iterations_  = scalar(&num_iterations_value_, types[3]);
        eps_value_            = static_cast<float>(1e-6);
        view_eps_             = scalar(&eps_value_, types[4]);
    }
    ~SinkhornBwdSession() override {
        hipFree(dgrad_output_);
        hipFree(dm_init_);
        hipFree(dgrad_logits_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainSinkhornBwd(view_grad_output_, view_m_init_, view_grad_logits_, view_num_iterations_,
                                      view_eps_, compute_, nullptr, 0, stream),
                    "ftrainSinkhornBwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_output_), "add grad_output");
        checkFtrain(ftrainPatternAddTensor(pattern, &tm_init_), "add m_init");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_logits_), "add grad_logits");
        checkFtrain(ftrainPatternAddTensor(pattern, &tnum_iterations_), "add num_iterations");
        checkFtrain(ftrainPatternAddTensor(pattern, &teps_), "add eps");
        checkFtrain(ftrainPatternAddSinkhornBwd(pattern, &operation_, tgrad_output_, tm_init_, tgrad_logits_,
                                                tnum_iterations_, teps_),
                    "ftrainPatternAddSinkhornBwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tgrad_output_, view_grad_output_), "set grad_output");
        checkFtrain(ftrainArgsSetTensor(args, tm_init_, view_m_init_), "set m_init");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_logits_, view_grad_logits_), "set grad_logits");
        checkFtrain(ftrainArgsSetTensor(args, tnum_iterations_, view_num_iterations_), "set num_iterations");
        checkFtrain(ftrainArgsSetTensor(args, teps_, view_eps_), "set eps");
        checkFtrain(ftrainArgsSetSinkhornBwd(args, operation_, compute_), "ftrainArgsSetSinkhornBwd");
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
    std::vector<std::int64_t> grad_output_dims_;
    void* dgrad_output_ = nullptr;
    std::vector<std::int64_t> m_init_dims_;
    void* dm_init_ = nullptr;
    std::vector<std::int64_t> grad_logits_dims_;
    void* dgrad_logits_ = nullptr;
    std::int64_t num_iterations_value_{};
    float eps_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_grad_output_{};
    FTrainTensorId tgrad_output_{};
    FTrainStorageView view_m_init_{};
    FTrainTensorId tm_init_{};
    FTrainStorageView view_grad_logits_{};
    FTrainTensorId tgrad_logits_{};
    FTrainStorageView view_num_iterations_{};
    FTrainTensorId tnum_iterations_{};
    FTrainStorageView view_eps_{};
    FTrainTensorId teps_{};
    FTrainSinkhornBwdOpId operation_{};
};
struct SinkhornBwdCase final : OpCase {
    const char* name() const override { return "sinkhorn_bwd"; }
    const char* shapeGrammar() const override { return "batch,matrix_size,iterations"; }
    const char* convenienceImpl() const override { return "ftrainSinkhornBwd"; }
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
                "batch,matrix_size,iterations: three positive values required");
        return std::make_unique<SinkhornBwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<SinkhornBwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
