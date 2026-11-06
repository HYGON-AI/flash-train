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

struct SinkhornFwdSession final : OpSession {
    SinkhornFwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        const auto a = shape[0], b = shape[1], c = shape[2];
        std::array<FTrainNumericType, 5> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_NUMERIC_TYPE_FP32};
                break;
            default: require(false, "unsupported precision");
        }
        logits_dims_ = {a, b, b};
        const std::size_t logits_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t logits_bytes = logits_count * elementBytes(types[0]);
        host_.resize((logits_bytes + 7) / 8);
        for (std::size_t j = 0; j < logits_count; ++j) { setHostValue(j, 0, types[0]); }
        checkHip(hipMalloc(&dlogits_, logits_bytes), "malloc logits");
        checkHip(hipMemcpy(dlogits_, host_.data(), logits_bytes, hipMemcpyHostToDevice), "fill logits");
        view_logits_ = dense(dlogits_, logits_dims_, types[0]);
        output_dims_ = {a, b, b};
        const std::size_t output_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t output_bytes = output_count * elementBytes(types[1]);
        host_.resize((output_bytes + 7) / 8);
        for (std::size_t j = 0; j < output_count; ++j) { setHostValue(j, 123, types[1]); }
        checkHip(hipMalloc(&doutput_, output_bytes), "malloc output");
        checkHip(hipMemcpy(doutput_, host_.data(), output_bytes, hipMemcpyHostToDevice), "fill output");
        view_output_ = dense(doutput_, output_dims_, types[1]);
        m_init_dims_ = {a, b, b};
        const std::size_t m_init_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(b);
        const std::size_t m_init_bytes = m_init_count * elementBytes(types[2]);
        host_.resize((m_init_bytes + 7) / 8);
        for (std::size_t j = 0; j < m_init_count; ++j) { setHostValue(j, 1.0 / b + 1e-6, types[2]); }
        checkHip(hipMalloc(&dm_init_, m_init_bytes), "malloc m_init");
        checkHip(hipMemcpy(dm_init_, host_.data(), m_init_bytes, hipMemcpyHostToDevice), "fill m_init");
        view_m_init_          = dense(dm_init_, m_init_dims_, types[2]);
        num_iterations_value_ = static_cast<std::int64_t>(c);
        view_num_iterations_  = scalar(&num_iterations_value_, types[3]);
        eps_value_            = static_cast<float>(1e-6);
        view_eps_             = scalar(&eps_value_, types[4]);
    }
    ~SinkhornFwdSession() override {
        hipFree(dlogits_);
        hipFree(doutput_);
        hipFree(dm_init_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainSinkhornFwd(view_logits_, view_output_, view_m_init_, view_num_iterations_, view_eps_,
                                      compute_, nullptr, 0, stream),
                    "ftrainSinkhornFwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tlogits_), "add logits");
        checkFtrain(ftrainPatternAddTensor(pattern, &toutput_), "add output");
        checkFtrain(ftrainPatternAddTensor(pattern, &tm_init_), "add m_init");
        checkFtrain(ftrainPatternAddTensor(pattern, &tnum_iterations_), "add num_iterations");
        checkFtrain(ftrainPatternAddTensor(pattern, &teps_), "add eps");
        checkFtrain(
            ftrainPatternAddSinkhornFwd(pattern, &operation_, tlogits_, toutput_, tm_init_, tnum_iterations_, teps_),
            "ftrainPatternAddSinkhornFwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tlogits_, view_logits_), "set logits");
        checkFtrain(ftrainArgsSetTensor(args, toutput_, view_output_), "set output");
        checkFtrain(ftrainArgsSetTensor(args, tm_init_, view_m_init_), "set m_init");
        checkFtrain(ftrainArgsSetTensor(args, tnum_iterations_, view_num_iterations_), "set num_iterations");
        checkFtrain(ftrainArgsSetTensor(args, teps_, view_eps_), "set eps");
        checkFtrain(ftrainArgsSetSinkhornFwd(args, operation_, compute_), "ftrainArgsSetSinkhornFwd");
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
    std::vector<std::int64_t> logits_dims_;
    void* dlogits_ = nullptr;
    std::vector<std::int64_t> output_dims_;
    void* doutput_ = nullptr;
    std::vector<std::int64_t> m_init_dims_;
    void* dm_init_ = nullptr;
    std::int64_t num_iterations_value_{};
    float eps_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_logits_{};
    FTrainTensorId tlogits_{};
    FTrainStorageView view_output_{};
    FTrainTensorId toutput_{};
    FTrainStorageView view_m_init_{};
    FTrainTensorId tm_init_{};
    FTrainStorageView view_num_iterations_{};
    FTrainTensorId tnum_iterations_{};
    FTrainStorageView view_eps_{};
    FTrainTensorId teps_{};
    FTrainSinkhornFwdOpId operation_{};
};
struct SinkhornFwdCase final : OpCase {
    const char* name() const override { return "sinkhorn_fwd"; }
    const char* shapeGrammar() const override { return "batch,matrix_size,iterations"; }
    const char* convenienceImpl() const override { return "ftrainSinkhornFwd"; }
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
        return std::make_unique<SinkhornFwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<SinkhornFwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
